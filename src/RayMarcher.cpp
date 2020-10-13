//-----------------------------------------------------------------------------
// Source code for ray marching demo
//
// Lot's stuff taken from
// https://www.iquilezles.org/www/index.htm
// http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
//-----------------------------------------------------------------------------

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <tchar.h>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "MathClasses.h"

//===================================================================================
// Options that you can enable or disable

// Allow the window to be resized
#define CAN_BE_RESIZED() 1

// Enable the console window to see debug output
#define ENABLE_CONSOLE() 0

// will update the preview buffer even if the scene has not finished
#define SHOW_RENDER_PROGRESS() 1

// draw an outline between objects and infinite space
#define DRAW_OBJECT_OUTLINE() 0

namespace
{
   // default settings that you can change

   // The default size of the render buffer
   uint32_t constexpr skDefaultWidth = 640;
   uint32_t constexpr skDefaultHeight = 480;

   // How long to wait between screen refreshes
   uint32_t constexpr skTimerMilliseconds = 100;

   // Setting this value smaller will speed up the application, but will prevent
   // things from very far in the distance from being rendered
   real32 constexpr skMaxLength = 60.f;

   // Setting this value bigger will speed up the application, but will make
   // the surfaces less accurate
   real32 constexpr skMinLength = 0.0001f;

   // how far off the surface to start a shadow or reflection ray
   real32 constexpr skSecondaryRayOffset = skMinLength * 10.f;


   // larger values will decrease render time, but make blocker
   uint32_t constexpr skInitialStepSize = 1;

   // Setting this number higher will make the UI more responsive for larger scenes
   // number of jobs to generate per core
#ifdef NDEBUG
   uint32_t constexpr skJobCoreMultiplier = 5;
#else
   uint32_t constexpr skJobCoreMultiplier = 50;
#endif

   // this is the color that will be used when missing the target
   CColor4f const skBackgroundColor( 0.2f, 0.3f, 0.4f );
}

//===================================================================================

namespace
{
   real32 constexpr skLargeNumber = 1e12f;
   real32 constexpr skSmallNumber = 1e-5f;
}

//===================================================================================

class SSurfaceInfo
{
public:
   real32 albedo{ 1.f };
   real32 metallic{ 0.f };
   real32 dielectric{ 0.f };
};

//-------------------------------------------------------------------------
// The texture objects

class CMaterialObject
{
public:
   using TPtr = std::shared_ptr<CMaterialObject>;
   using TConstPtr = std::shared_ptr<CMaterialObject const>;

   virtual ~CMaterialObject() = 0 { }

   void SetTransform( CTransform4f const& transform )
   {
      mTransform = transform;
      mInverseTransform = mTransform.GetInverse();
   }

   CTransform4f const& GetTransform() const
   {
      return mTransform;
   }

   CColor4f GetTransformedColorAtPoint( CVector3f const& point ) const
   {
      return GetColorAtPoint( mInverseTransform * point );
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const = 0;

private:
   CTransform4f mTransform{ CTransform4f::Identity() };
   CTransform4f mInverseTransform{ CTransform4f::Identity() };
};

class CMaterialContainer
{
public:
   explicit CMaterialContainer( CMaterialObject* pMaterialObject )
      : mMaterialObject( pMaterialObject )
   {
   }

   CMaterialObject::TConstPtr GetMaterial() const
   {
      return mMaterialObject;
   }

   CMaterialContainer& operator<<( CTransform4f const& transform )
   {
      mMaterialObject->SetTransform( transform );
      return *this;
   }

private:
   CMaterialObject::TPtr mMaterialObject;
};

class CColorMaterialObject : public CMaterialObject
{
public:
   explicit CColorMaterialObject( CColor4f const& color )
      : mColor( color )
   {
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& /*point*/ ) const override
   {
      return mColor;
   }

private:
   CColor4f mColor;
};

class CCheckerMaterialObject : public CMaterialObject
{
public:
   explicit CCheckerMaterialObject( CColor4f const& color0, CColor4f const & color1 )
      : mColor0( color0 ), mColor1( color1 )
   {
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const override
   {
      int32_t const sum = 
         static_cast<int32_t>(floorf(point.GetX())) + 
         static_cast<int32_t>(floorf(point.GetY())) +
         static_cast<int32_t>(floorf(point.GetZ()));
      if ((sum & 1) == 0)
      {
         return mColor0;
      }
      return mColor1;
   }

private:
   CColor4f mColor0;
   CColor4f mColor1;
};

class CGradientMaterialObject : public CMaterialObject
{
public:
   explicit CGradientMaterialObject( CColor4f const& color0, CColor4f const& color1 )
      : mColor0( color0 ), mColor1( color1 )
   {
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const override
   {
      real32 const distanceFromCenter = point.Magnitude();

      real32 const phase = distanceFromCenter - floorf( distanceFromCenter );

      return CColor4f::Lerp( mColor0, mColor1, phase );
   }

private:
   CColor4f mColor0;
   CColor4f mColor1;
};

//-------------------------------------------------------------------------

class CCustomMaterialObject : public CMaterialObject
{
public:
   using TCallback = std::function<CColor4f( CVector3f )>;
   explicit CCustomMaterialObject( TCallback const& customFunction )
      : mCustomFunction( customFunction )
   {
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const override
   {
      return mCustomFunction( point );
   }

private:
   TCallback mCustomFunction;
};

//-------------------------------------------------------------------------

// The render objects

class CRenderObject
{
public:
   using TPtr = std::shared_ptr<CRenderObject>;
   using TConstPtr = std::shared_ptr<CRenderObject const>;

   virtual ~CRenderObject() = 0 { }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const = 0;

   real32 GetTransformedDistanceToPoint( CVector3f const& point ) const
   {
      return GetDistanceToPoint( mInverseTransform * point );
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const
   {
      if (mMaterial.get() != nullptr)
      {
         return mMaterial->GetTransformedColorAtPoint( mInverseTransform * point );
      }
      return CColor4f::White();
   }

   virtual void SetMaterial( CMaterialObject::TConstPtr const& material )
   {
      mMaterial = material;
   }

   void SetTransform( CTransform4f const& transform )
   {
      mTransform = transform;
      mInverseTransform = mTransform.GetInverse();
   }

   CTransform4f const& GetTransform() const
   {
      return mTransform;
   }

   CTransform4f const& GetInverseTransform() const
   {
      return mInverseTransform;
   }

   SSurfaceInfo const& GetSurfaceInfo() const
   {
      return mSurfaceInfo;
   }

   void SetSurfaceInfo( SSurfaceInfo const& surfaceInfo )
   {
      mSurfaceInfo = surfaceInfo;
   }

private:
   CMaterialObject::TConstPtr mMaterial;
   CTransform4f mTransform{ CTransform4f::Identity() };
   CTransform4f mInverseTransform{ CTransform4f::Identity() };
   SSurfaceInfo mSurfaceInfo;
};

using TRenderObjects = std::vector< CRenderObject* >;

//-------------------------------------------------------------------------

// this is a wrapper around objects to make them easier to 
// manipulate during creation

class CObjectContainer
{
public:

   explicit CObjectContainer( CRenderObject* pRenderObject )
      : mpRenderObject( pRenderObject )
   {
   }

   CObjectContainer& operator<<( CTransform4f const& transform )
   {
      mpRenderObject->SetTransform( transform );
      return *this;
   }

   CObjectContainer& operator<<( CColor4f const& color )
   {
      mpRenderObject->SetMaterial( CMaterialObject::TConstPtr( new CColorMaterialObject( color ) ) );
      return *this;
   }

   CObjectContainer& operator<<( CMaterialContainer const& material )
   {
      mpRenderObject->SetMaterial( material.GetMaterial() );
      return *this;
   }

   CObjectContainer& operator<<( SSurfaceInfo const& surfaceInfo )
   {
      mpRenderObject->SetSurfaceInfo( surfaceInfo );
      return *this;
   }

   CRenderObject::TPtr GetRenderObject() const
   {
      return mpRenderObject;
   }

private:
   std::shared_ptr< CRenderObject > mpRenderObject;
};

using TObjectContainers = std::vector< CObjectContainer >;

//-----------------------------------------------------------------------------------

// Lots of distance functions
// https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

class CRenderSphere : public CRenderObject
{
public:
   explicit CRenderSphere( CVector3f const& center, real32 const radius )
      : mCenter( center )
      , mRadius( radius )
   {
   }
   explicit CRenderSphere( real32 const radius )
      : mCenter( CVector3f::Zero() )
      , mRadius( radius )
   {
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      return (point - mCenter).Magnitude() - mRadius;
   }

private:
   CVector3f mCenter;
   real32 mRadius;
};

//-----------------------------------------------------------------------------

class CRenderPlane : public CRenderObject
{
public:
   explicit CRenderPlane( CVector3f const& normal, real32 const height )
      : mNormal( normal )
      , mHeight( height )
   {
   }

   explicit CRenderPlane( CVector3f const& normal )
      : mNormal( normal )
      , mHeight( 0 )
   {
   }

   virtual real32 GetDistanceToPoint( CVector3f const & point ) const override
   {
      return CVector3f::Dot( mNormal, point ) - mHeight;
   }

private:
   CVector3f mNormal;
   real32 mHeight;
};

//-----------------------------------------------------------------------------

class CRenderCube : public CRenderObject
{
public:
   explicit CRenderCube( CVector3f const& size )
      : mSize( size * 0.5f )
   {
   }

   explicit CRenderCube( real32 const x, real32 const y, real32 const z )
      : mSize( x * 0.5f, y * 0.5f, z * 0.5f )
   {
   }

   explicit CRenderCube( real32 const size )
      : mSize( size * 0.5f, size * 0.5f, size * 0.5f  )
   {
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      real32 const x = NMath::AbsF( point.GetX() ) - mSize.GetX();
      real32 const y = NMath::AbsF( point.GetY() ) - mSize.GetY();
      real32 const z = NMath::AbsF( point.GetZ() ) - mSize.GetZ();
      // distance outside
      real32 const d = CVector3f( NMath::max_val( x, 0.f ), NMath::max_val( y, 0.f ), NMath::max_val( z, 0.f ) ).Magnitude();
      // distance inside
      real32 const du = NMath::max_val( NMath::max_val( NMath::min_val( x, 0.f ), NMath::min_val( y, 0.f ) ), NMath::min_val( z, 0.f ) );
      return  d + du;
   }

private:
   CVector3f mSize;
};

//-----------------------------------------------------------------------------

class CRenderCustom : public CRenderObject
{
public:
   using TCallback = std::function<real32( CVector3f )>;
   explicit CRenderCustom( TCallback const& customFunction )
      : mCustomFunction( customFunction )
   {
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      return mCustomFunction( point );
   }

private:
   TCallback mCustomFunction;
};

//-----------------------------------------------------------------------------

class CCompositeRenderObject : public CRenderObject
{
public:
   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const override
   {
#if 0
      CRenderObject::TConstPtr closestObject;
      real32 minPoint = skLargeNumber;

      for (CRenderObject::TConstPtr const & object : mObjectList)
      {
         real32 const objectDistance = NMath::AbsF( object->GetTransformedDistanceToPoint( GetInverseTransform() * point ) );
         if ( objectDistance < minPoint)
         {
            minPoint = objectDistance;
            closestObject = object;
         }
      }
      if (closestObject.get() != nullptr)
      {
         return closestObject->GetColorAtPoint( GetInverseTransform() * point );
      }
      return CColor4f::White();
#else

      CColor4f color = CColor4f::Black();

      real32 weight = 0.f;

      for (CRenderObject::TConstPtr const& object : mObjectList)
      {
         real32 const objectDistance = NMath::AbsF( object->GetTransformedDistanceToPoint( GetInverseTransform() * point ) );

         CColor4f const objectColor = object->GetColorAtPoint( GetInverseTransform() * point );
         if (NMath::small_enough( objectDistance ))
         {
            return objectColor;
         }
         real32 const objectWeight = 1.f / powf( objectDistance , 0.9f );
         weight += objectWeight;
         color = color + objectColor * objectWeight;
      }

      return color * (1.f / weight);

#endif
   }

   virtual void SetMaterial( CMaterialObject::TConstPtr const& material )
   {
      for (CRenderObject::TPtr & object : mObjectList)
      {
         object->SetMaterial( material );
      }
   }

protected:
   std::vector< CRenderObject::TPtr > mObjectList;

};

//-----------------------------------------------------------------------------

class CRenderUnion : public CCompositeRenderObject
{
public:
   explicit CRenderUnion( std::initializer_list<CObjectContainer> const & objects )
   {
      mObjectList.reserve( objects.size() );
      for (CObjectContainer const& object : objects)
      {
         mObjectList.push_back( CRenderObject::TPtr( object.GetRenderObject() ) );
      }
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      real32 minValue = skLargeNumber;

      for (CRenderObject::TPtr const& object : mObjectList)
      {
         minValue = NMath::min_val( minValue, object->GetTransformedDistanceToPoint( point ) );
      }

      return minValue;
   }
};

//-----------------------------------------------------------------------------

class CRenderIntersection : public CCompositeRenderObject
{
public:
   explicit CRenderIntersection( std::initializer_list<CObjectContainer> const & objects )
   {
      mObjectList.reserve( objects.size() );
      for (CObjectContainer const& object : objects)
      {
         mObjectList.push_back( CRenderObject::TPtr( object.GetRenderObject() ) );
      }
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      real32 maxValue = 0.f;

      for (CRenderObject::TPtr const& object : mObjectList)
      {
         maxValue = NMath::max_val( maxValue, object->GetTransformedDistanceToPoint( point ) );
      }

      return maxValue;
   }
};

//-----------------------------------------------------------------------------

class CRenderDifference : public CCompositeRenderObject
{
public:
   explicit CRenderDifference( std::initializer_list<CObjectContainer> const& objects )
   {
      mObjectList.reserve( objects.size() );
      for (CObjectContainer const& object : objects)
      {
         mObjectList.push_back( CRenderObject::TPtr( object.GetRenderObject() ) );
      }
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      real32 maxValue = 0.f;
      int32_t index = 0;
      for (CRenderObject::TPtr const& object : mObjectList)
      {
         // the first object is normal and all other objects cut from it
         real32 const distance = (index++ ? -1.f : 1.f) * object->GetTransformedDistanceToPoint( point );
         maxValue = NMath::max_val( maxValue, distance );
      }

      return maxValue;
   }
};

//-----------------------------------------------------------------------------

class CRenderSmoothUnion : public CCompositeRenderObject
{
public:
   explicit CRenderSmoothUnion( std::initializer_list<CObjectContainer> const& objects, real32 const k )
      : mK(k)
   {
      mObjectList.reserve( objects.size() );
      for (CObjectContainer const& object : objects)
      {
         mObjectList.push_back( CRenderObject::TPtr( object.GetRenderObject() ) );
      }
   }

   static real32 const SmoothUnion( real32 const d1, real32 const d2, real32 const k )
   {
      float const h = NMath::max_val( k - NMath::AbsF( d1 - d2 ), 0.f ) / k;
      return NMath::min_val( d1, d2 ) - h * h * h * k * (1.f / 6.f);
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      real32 minValue = skLargeNumber;

      int32_t index = 0;

      for (CRenderObject::TPtr const& object : mObjectList)
      {
         if (index++ == 0)
         {
            minValue = object->GetTransformedDistanceToPoint( point );
         }
         else
         {
            minValue = SmoothUnion( minValue, object->GetTransformedDistanceToPoint( point ), mK );
         }
      }

      return minValue;
   }
private:
   real32 mK;
};

class CRenderBlend : public CCompositeRenderObject
{
public:
   explicit CRenderBlend( std::initializer_list<CObjectContainer> const& objects, real32 const k )
      : mK( k )
   {
      mObjectList.reserve( objects.size() );
      for (CObjectContainer const& object : objects)
      {
         mObjectList.push_back( CRenderObject::TPtr( object.GetRenderObject() ) );
      }
   }

   virtual real32 GetDistanceToPoint( CVector3f const& point ) const override
   {
      uint32_t const lowerPosition = static_cast<uint32_t>(floorf( mK ));
      uint32_t const upperPosition = lowerPosition + 1;

      real32 const d0 = lowerPosition >= 0 && lowerPosition < mObjectList.size() ?
         mObjectList[lowerPosition]->GetTransformedDistanceToPoint( GetInverseTransform()*point ) : skLargeNumber;
      real32 const d1 = upperPosition >= 0 && upperPosition < mObjectList.size() ?
         mObjectList[upperPosition]->GetTransformedDistanceToPoint( GetInverseTransform()*point ) : skLargeNumber;
      return NMath::lerp(d0,d1, mK - floorf(mK));
   }

   virtual CColor4f const GetColorAtPoint( CVector3f const& point ) const override
   {
      uint32_t const lowerPosition = static_cast<uint32_t>(floorf( mK ));
      uint32_t const upperPosition = lowerPosition + 1;

      CColor4f const c0 = lowerPosition >= 0 && lowerPosition < mObjectList.size() ?
         mObjectList[lowerPosition]->GetColorAtPoint( GetInverseTransform() * point ) : CColor4f::Black();
      CColor4f const c1 = upperPosition >= 0 && upperPosition < mObjectList.size() ?
         mObjectList[upperPosition]->GetColorAtPoint( GetInverseTransform() * point ) : CColor4f::Black();

      return CColor4f::Lerp( c0, c1, mK - floorf( mK ) );
   }
private:
   real32 mK;
};


//-----------------------------------------------------------------------------

class CLightObject
{
public:
   using TPtr = std::shared_ptr< CLightObject >;
   using TConstPtr = std::shared_ptr< CLightObject const >;

   virtual ~CLightObject() = 0 {}

   virtual CColor4f CalculateValueAtPosition( CVector3f const& position, CVector3f const& surfaceNormal ) const = 0;
   virtual CVector3f const& GetPosition() const = 0;
   virtual bool CastsShadow() const = 0;
};

//-----------------------------------------------------------------------------

class CAmbientLightObject : public CLightObject
{
public:
   explicit CAmbientLightObject( CColor4f const& color )
      : mColor( color )
   {
   }

   virtual CColor4f CalculateValueAtPosition( CVector3f const& /*position*/, CVector3f const& /*surfaceNormal*/ ) const override
   {
      return mColor;
   }

   virtual CVector3f const& GetPosition() const override
   {
      static CVector3f const skZero( CVector3f::Zero() );
      return skZero;
   }

   virtual bool CastsShadow() const
   {
      return false;
   }

private:
   CColor4f mColor;
};

//-----------------------------------------------------------------------------

class CPointLightObject : public CLightObject
{
public:
   explicit CPointLightObject( CVector3f const& position, CColor4f const& color )
      : mPosition( position )
      , mColor( color )
   {
   }

   virtual CColor4f CalculateValueAtPosition( CVector3f const& position, CVector3f const & surfaceNormal ) const override
   {
      CVector3f const direction = (mPosition - position).AsNormalized();
      real32 const angle = CVector3f::Dot( surfaceNormal, direction );
      if (angle < 0.f)
      {
         return CColor4f::Black();
      }
      return  mColor * angle;
   }

   virtual CVector3f const& GetPosition() const override
   {
      return mPosition;
   }

   virtual bool CastsShadow() const override
   {
      return true;
   }

   CColor4f const& GetColor() const
   {
      return mColor;
   }

private:
   CVector3f mPosition;
   CColor4f mColor;
};

class CDirectionalLightObject : public CLightObject
{
public:
   explicit CDirectionalLightObject( CVector3f const& direction, CColor4f const& color )
      : mDirection( direction.AsNormalized() )
      , mColor( color )
   {
   }

   virtual CColor4f CalculateValueAtPosition( CVector3f const& /*position*/, CVector3f const& surfaceNormal ) const override
   {
      real32 const angle = CVector3f::Dot( surfaceNormal, mDirection );
      if (angle < 0.f)
      {
         return CColor4f::Black();
      }
      return  mColor * angle;
   }

   virtual CVector3f const& GetPosition() const override
   {
      static CVector3f const skZero( CVector3f::Zero() );
      return skZero;
   }

   virtual bool CastsShadow() const override
   {
      return false;
   }

   CColor4f const& GetColor() const
   {
      return mColor;
   }

private:
   CVector3f mDirection;
   CColor4f mColor;
};

//===================================================================================

class CInfiniteRay
{
public:
   explicit CInfiniteRay( CVector3f const& position, CVector3f const& direction ) : mPosition( position ), mDirection( direction ) { }
   CVector3f const& GetPosition() const { return mPosition; }
   CVector3f const& GetDirection() const { return mDirection; }
   CVector3f const GetPositionAlongRay( real32 const time ) const { return mPosition + mDirection * time; }
private:
   CVector3f mPosition;
   CVector3f mDirection;
};

//-------------------------------------------------------------------------

class CRayResult
{
public:
   explicit CRayResult( CVector3f const& collisionPoint, real32 const time, bool const hit )
      : mCollisionPoint( collisionPoint )
      , mTime( time )
      , mHit( hit )
   {
   }

   static CRayResult NoResults() { return CRayResult( CVector3f::Zero(), skLargeNumber, false ); }

   CVector3f mCollisionPoint;
   real32 mTime;
   bool mHit;
};

//-------------------------------------------------------------------------

class CCamera
{
public:
   explicit CCamera( CVector3f const& cameraCenter, CVector3f const& cameraLookAt, real32 const cameraFOV = 45.f, bool const verticalFOV = false )
   : mCameraTransform( CTransform4f::Identity())
   , mSceneWidth(skDefaultWidth)
   , mSceneHeight(skDefaultHeight)
   , mCameraScale(1.f)
   , mCameraFOV(cameraFOV)
   , mVerticalFOV(verticalFOV)
   {
      CalculateParameters(cameraCenter, cameraLookAt, cameraFOV, verticalFOV);
   }

   static CCamera DefaultCamera() { return CCamera( CVector3f::Zero(), CVector3f( 0.f, 0.f, 1.f ), 45.f ); }

   CInfiniteRay const GetRayForPosition( uint32_t const x, uint32_t const y ) const
   {
      real32 const hFactor = (x - (mSceneWidth * 0.5f)) * mCameraScale;
      real32 const vFactor = -(y - (mSceneHeight * 0.5f)) * mCameraScale;

      CVector3f const direction = mCameraTransform.GetZBasis() + mCameraTransform.GetXBasis() * hFactor + mCameraTransform.GetYBasis() * vFactor;

      return CInfiniteRay( mCameraTransform.GetTranslation(), direction.AsNormalized() );
   }

   real32 const GetCameraScale() const { return mCameraScale; }
   CTransform4f const& GetCameraTransform() const { return mCameraTransform; }
   void SetCameraTransform( CTransform4f const& transform ) { mCameraTransform = transform; }

   void SetSceneSize( uint32_t const sceneWidth, uint32_t const sceneHeight )
   {
      mSceneWidth = sceneWidth;
      mSceneHeight = sceneHeight;

      CalculateParameters( mCameraTransform.GetTranslation(), mCameraTransform.GetTranslation() + mCameraTransform.GetBackward(), mCameraFOV, mVerticalFOV );
   }

private:

   void CalculateParameters( CVector3f const& cameraCenter, CVector3f const& cameraLookAt, real32 const cameraFOV, bool const verticalFOV )
   {
      CVector3f const worldUp = CVector3f::Up();

      real32 const fovScale( tanf( cameraFOV * ((NMath::gkPi32 / 180.f) / 2.f) ) * 2.f );

      if (verticalFOV)
      {
         mCameraScale = fovScale / mSceneHeight;
      }
      else
      {
         mCameraScale = fovScale / mSceneWidth;
      }

      CVector3f const cameraForward = (cameraLookAt - cameraCenter).AsNormalized();
      CVector3f const cameraRight = cameraForward.CrossProductWith( worldUp ).AsNormalized();
      CVector3f const cameraUp = cameraRight.CrossProductWith( cameraForward );

      mCameraTransform = CTransform4f::FromVectors( cameraRight, cameraUp, cameraForward, cameraCenter );
   }

   CTransform4f mCameraTransform;
   uint32_t mSceneWidth;
   uint32_t mSceneHeight;
   real32 mCameraScale;
   real32 mCameraFOV;
   bool mVerticalFOV;
};

//-------------------------------------------------------------------------

class CLightObjectContainer
{
public:

   explicit CLightObjectContainer( CLightObject* pLightObject )
      : mpLightObject( pLightObject )
   {
   }

   CLightObject const * GetLightObject() const
   {
      return mpLightObject;
   }

private:
   CLightObject* mpLightObject;
};

//-------------------------------------------------------------------------

class CRenderScene
{
public:
   explicit CRenderScene( TRenderObjects const& renderObjects )
      : mCamera( CCamera::DefaultCamera() )
   {
      mObjects.reserve( renderObjects.size() );
      for (CRenderObject const* const object : renderObjects)
      {
mObjects.push_back( CRenderObject::TConstPtr( object ) );
      }
   }

   explicit CRenderScene()
      : mCamera( CCamera::DefaultCamera() )
   {
   }

   CRenderScene& operator+=( CObjectContainer const& containerObject )
   {
      mObjects.push_back( CRenderObject::TConstPtr( containerObject.GetRenderObject() ) );
      return *this;
   }

   CRenderScene& operator+=( CLightObjectContainer const& containerObject )
   {
      mLights.push_back( CLightObject::TConstPtr( containerObject.GetLightObject() ) );
      return *this;
   }

   void AddObject( CRenderObject* pRenderObject )
   {
      mObjects.push_back( CRenderObject::TConstPtr( pRenderObject ) );
   }

   CRenderScene& operator<<( CCamera const& camera )
   {
      mCamera = camera;
      return *this;
   }

   CRenderScene& operator<<( CTransform4f const& cameraTransform )
   {
      mCamera.SetCameraTransform( cameraTransform );
      return *this;
   }

   CRenderScene& operator*( CTransform4f const& cameraTransform )
   {
      mCamera.SetCameraTransform( cameraTransform * mCamera.GetCameraTransform() );
      return *this;
   }

   void SetCamera( CCamera const& camera )
   {
      mCamera = camera;
   }


   void SetSceneSize( uint32_t const width, uint32_t const height )
   {
      mCamera.SetSceneSize( width, height );
   }

   CColor4f DoIntersection( uint32_t const x, uint32_t const y ) const
   {
      CInfiniteRay const infiniteRay = mCamera.GetRayForPosition( x, y );
      return DoIntersection( infiniteRay, 4 );
   }

   //----------------------------------------------------------------------------

   CColor4f DoIntersection( CInfiniteRay const& infiniteRay, int32_t const depth = 1 ) const
   {
      if (depth == 0)
      {
         return CColor4f::Black();
      }

      CRayResult result = MarchRay( infiniteRay, skMaxLength );

      if (result.mHit)
      {
         CRenderObject const* const pRenderObject = GetClosestObject( result.mCollisionPoint );
         if (pRenderObject != nullptr)
         {
            return CalculateSurfaceColor( pRenderObject, infiniteRay.GetDirection(), result.mCollisionPoint, depth );
         }
      }
#if DRAW_OBJECT_OUTLINE()
      else
      {
         if (result.mTime < 0.05f)
         {
            return CColor4f::Lerp( CColor4f::White(), skBackgroundColor, result.mTime * 20.f);
         }
      }
#endif

      return skBackgroundColor;
   }

   //----------------------------------------------------------------------------

   CColor4f CalculateSurfaceColor( CRenderObject const* const pRenderObject, CVector3f const & viewDirection, CVector3f const& collisionPoint, int32_t const depth ) const
   {
      CColor4f color = CColor4f::Black();

      CVector3f const normal = GetNormalAtPoint( collisionPoint );


      CColor4f const surfaceColor = pRenderObject->GetColorAtPoint( collisionPoint );

      // get the start of the ray off of the surface just a little bit
      CVector3f const startPoint = collisionPoint + normal * skSecondaryRayOffset;

      SSurfaceInfo const surfaceInfo = pRenderObject->GetSurfaceInfo();

      if (!NMath::small_enough( surfaceInfo.dielectric ) || !NMath::small_enough( surfaceInfo.metallic ))
      {
         CVector3f const reflection = viewDirection - normal * 2.f * CVector3f::Dot( viewDirection, normal );
         CColor4f const reflectedColor = DoIntersection( CInfiniteRay( startPoint, reflection ), depth - 1 );

         color += reflectedColor * surfaceColor * surfaceInfo.metallic;
         color += reflectedColor * surfaceInfo.dielectric;
      }

      // for each light do a light check
      for (CLightObject::TConstPtr const& pLight : mLights)
      {
         CVector3f toLight = pLight->GetPosition() - collisionPoint;
         real32 const distance = toLight.Magnitude();
         toLight = toLight / distance;

         if (pLight->CastsShadow())
         {
            real32 const shadow = MarchShadowRay( CInfiniteRay( startPoint, toLight ), distance, 24.f );

            if (shadow > 0.f)
            {
               color += pLight->CalculateValueAtPosition( collisionPoint, normal ) * surfaceColor 
                  * (shadow * surfaceInfo.albedo);
            }
         }
         else
         {
            color += pLight->CalculateValueAtPosition( collisionPoint, normal ) * 
               ( surfaceColor * surfaceInfo.albedo );
         }
      }
      return color;
   }

   //----------------------------------------------------------------------------
   // This is the marching ray code

   CRayResult MarchRay( CInfiniteRay const& ray, real32 const maxLength ) const
   {
      real32 time = skMinLength;

      int32_t count = 0;
      real32 minDistance = skLargeNumber;

      while (time < maxLength  )
      {
         CVector3f const currentPoint = ray.GetPositionAlongRay( time );
         real32 const distanceToNearestObject = GetMinDistanceAtPoint( currentPoint );
         minDistance = NMath::min_val( minDistance, distanceToNearestObject );

         if ( fabsf(distanceToNearestObject) < skMinLength || count++ > 200)
         {
            return CRayResult( currentPoint, time, true );
         }

         time += distanceToNearestObject;
      }
      return CRayResult(CVector3f::Zero(), minDistance, false );
   }

   //----------------------------------------------------------------------------
   // Calculate the shadow amount
   // See: https://iquilezles.org/www/articles/rmshadows/rmshadows.htm

   real32 MarchShadowRay( CInfiniteRay const& ray, real32 const maxLength, real32 const penumbra ) const
   {
#if 1
      real32 shadow = 1.f;
      real32 time = 0.f;

      while (time < maxLength )
      {
         CVector3f const currentPoint = ray.GetPositionAlongRay( time );
         real32 const distanceToNearestObject = GetMinDistanceAtPoint( currentPoint );

         if (distanceToNearestObject < skMinLength )
         {
            return 0.f;
         }

         shadow = NMath::min_val( shadow, penumbra * distanceToNearestObject / time );

         time += distanceToNearestObject;
      }
      return shadow;
#else
      real32 shadow = 1.f;
      real32 ph = skLargeNumber;

      real32 time = skMinLength * 2.f;

      int32_t count = 0;
      while (time < maxLength && count++ < 25)
      {
         CVector3f const currentPoint = ray.GetPositionAlongRay( time );
         real32 const distanceToNearestObject = GetMinDistanceAtPoint( currentPoint );

         if (distanceToNearestObject < 0.f)
         {
            return 0.f;
         }

         real32 const y = distanceToNearestObject * distanceToNearestObject / (2.f * ph);
         real32 const d = sqrtf( distanceToNearestObject * distanceToNearestObject - y * y );
         shadow = NMath::min_val( shadow, penumbra * d / NMath::max_val( 0.f, time - y ) );
         ph = distanceToNearestObject;
         time += distanceToNearestObject;
      }

      return shadow;
#endif
   }

   //----------------------------------------------------------------------------

   CVector3f GetNormalAtPoint( CVector3f const& point ) const
   {
#if 1
      //real32 const skNormalEpsilon = 0.1f * (1.f / skDefaultWidth);
      real32 const skNormalEpsilon = skSecondaryRayOffset;
      return
      // look at the gradient in the local area
      CVector3f( GetMinDistanceAtPoint( point + CVector3f( skNormalEpsilon, 0.f, 0.f ) ) - GetMinDistanceAtPoint( point - CVector3f( skNormalEpsilon, 0.f, 0.f ) ),
                 GetMinDistanceAtPoint( point + CVector3f( 0.f, skNormalEpsilon, 0.f ) ) - GetMinDistanceAtPoint( point - CVector3f( 0.f, skNormalEpsilon, 0.f ) ),
                 GetMinDistanceAtPoint( point + CVector3f( 0.f, 0.f, skNormalEpsilon ) ) - GetMinDistanceAtPoint( point - CVector3f( 0.f, 0.f, skNormalEpsilon ) ) ).AsNormalized();
#else

      real32 const skNormalEpsilon = 0.5773f * (1.f / skDefaultWidth);

      CVector3f const e0( 1.f, -1.f, -1.f );
      CVector3f const e1( -1.f, -1.f, 1.f );
      CVector3f const e2( -1.f, 1.f, -1.f );
      CVector3f const e3( 1.f, 1.f, 1.f );

      return     
         (e0 * GetMinDistanceAtPoint( point + e0 * skNormalEpsilon )  +
        e1 * GetMinDistanceAtPoint( point + e1 * skNormalEpsilon ) +
        e2 * GetMinDistanceAtPoint( point + e2 * skNormalEpsilon ) +
        e3 * GetMinDistanceAtPoint( point + e3 * skNormalEpsilon)).AsNormalized();

#endif
   }

   //----------------------------------------------------------------------------

   real32 GetMinDistanceAtPoint( CVector3f const& point ) const
   {
      real32 time = skLargeNumber;

      for (CRenderObject::TConstPtr const & pObject : mObjects)
      {
         time = NMath::min_val( time, pObject->GetTransformedDistanceToPoint( point ) );
      }

      return time;
   }

   //----------------------------------------------------------------------------

   CRenderObject const * GetClosestObject( CVector3f const& point ) const
   {
      real32 minTime = skLargeNumber;
      CRenderObject const* pClosestObject = nullptr;
      for (CRenderObject::TConstPtr const & pObject : mObjects)
      {
         real32 const currentTime = pObject->GetTransformedDistanceToPoint( point );

         if (currentTime < minTime)
         {
            minTime = currentTime;
            pClosestObject = pObject.get();
         }
      }
      return pClosestObject;
   }

   void Reset()
   {
      mCamera = CCamera::DefaultCamera();
      mObjects.clear();
      mLights.clear();
   }

private:
   CCamera mCamera;
   std::vector< CRenderObject::TConstPtr > mObjects;
   std::vector< CLightObject::TConstPtr > mLights;
};

//===================================================================================

namespace NScene
{
   template<class TClassType>
   class TMaterialContainer : public CMaterialContainer
   {
   public:
      template<class... TArgs>
      explicit TMaterialContainer( TArgs... args )
         : CMaterialContainer( new TClassType( args... ) )
      {
      }
   };

   template<class TClassType>
   class TObjectContainer : public CObjectContainer
   {
   public:
      template <class... TArgs>
      explicit TObjectContainer( TArgs... args )
         : CObjectContainer( new TClassType( args... ) )
      {
      }
      
      template <class... TArgs>
      TObjectContainer( std::initializer_list<CObjectContainer> const& objects, TArgs... args )
         : CObjectContainer( new TClassType( objects, args... ) )
      {
      }
      
   };

   template<class TClassType>
   class TLightObjectContainer : public CLightObjectContainer
   {
   public:
      template <class... TArgs>
      explicit TLightObjectContainer( TArgs... args )
         : CLightObjectContainer( new TClassType( args... ) )
      {
      }
   };

   // convenience types
   using vector3 = CVector3f;
   using color = CColor4f;

   // camera
   using camera = CCamera;

   // materials
   using surface = SSurfaceInfo;
   using material = CMaterialContainer;
   using checker = TMaterialContainer< CCheckerMaterialObject >;
   using gradient = TMaterialContainer< CGradientMaterialObject >;
   using custom_material = TMaterialContainer< CCustomMaterialObject >;

   // objects
   using object = CObjectContainer;
   using sphere = TObjectContainer<CRenderSphere>;
   using plane = TObjectContainer<CRenderPlane>;
   using cube = TObjectContainer<CRenderCube>;
   using custom = TObjectContainer<CRenderCustom>;

   // csg operations
   using csg_union = TObjectContainer<CRenderUnion>;
   using csg_intersection = TObjectContainer<CRenderIntersection>;
   using csg_difference = TObjectContainer<CRenderDifference>;
   using csg_smoothunion = TObjectContainer<CRenderSmoothUnion>;

   using blend = TObjectContainer<CRenderBlend>;

   // lights
   using ambientlight = TLightObjectContainer<CAmbientLightObject>;
   using pointlight = TLightObjectContainer<CPointLightObject>;
   using directionallight = TLightObjectContainer<CDirectionalLightObject>;
   

   // convenience functions
   CTransform4f translate( real32 const x, real32 const y, real32 const z ) { return CTransform4f::Translate( x, y, z ); }
   CTransform4f translate( CVector3f const & translation ) { return CTransform4f::Translate( translation ); }
   CTransform4f scale( real32 const x, real32 const y, real32 const z ) { return CTransform4f::Scale( x, y, z ); }
   CTransform4f scale( CVector3f const& transformScale ) { return CTransform4f::Scale( transformScale ); }
   CTransform4f scale( real32 const value ) { return CTransform4f::Scale( value, value, value ); }
   CTransform4f rotatex( real32 const angle ) { return CTransform4f::RotateX( CRelAngle::FromDegrees( angle ) ); }
   CTransform4f rotatey( real32 const angle ) { return CTransform4f::RotateY( CRelAngle::FromDegrees( angle ) ); }
   CTransform4f rotatez( real32 const angle ) { return CTransform4f::RotateZ( CRelAngle::FromDegrees( angle ) ); }
   CTransform4f rotate( real32 const x, real32 const y, real32 const z) 
   { 
      return
         CTransform4f::RotateX( CRelAngle::FromDegrees( x ) ) *
         CTransform4f::RotateY( CRelAngle::FromDegrees( y ) ) *
         CTransform4f::RotateZ( CRelAngle::FromDegrees( z ) );
   }

   real32 length( CVector3f const& v ) { return v.Magnitude(); }
   CVector3f normalize( CVector3f const& v ) { return v.AsNormalized(); }

   real32 minf( real32 const lhs, real32 const rhs ) { return lhs < rhs ? lhs : rhs; }
   real32 maxf( real32 const lhs, real32 const rhs ) { return lhs > rhs ? lhs : rhs; }
   real32 round_mod( real32 const x, real32 const y ) { return x - y * roundf( x / y ); }
   real32 mod( real32 const x, real32 const y ) { return x - y * floorf( x / y ); }
   real32 clamp( real32 const minValue, real32 const value, real32 const maxValue ) { return NMath::clamp( minValue, value, maxValue ); }
   CVector3f clamp( CVector3f const & minValue, CVector3f const value, CVector3f const & maxValue ) 
   { return CVector3f( clamp( minValue.x, value.x, maxValue.x ), clamp( minValue.y, value.y, maxValue.y ), clamp( minValue.z, value.z, maxValue.z ) ); }

   real32 dot( CVector3f const& lhs, CVector3f const& rhs ) { return CVector3f::Dot( lhs, rhs ); }
   CVector3f cross( CVector3f const& lhs, CVector3f const& rhs ) { return CVector3f::Cross( lhs, rhs ); }




   void BuildScene( CRenderScene& scene, real32 const time )
   {
      UNREFERENCED_PARAMETER( time );
      UNREFERENCED_PARAMETER( scene );

#include "RenderScene.inl"
   }
}

//===================================================================================
// This class coordinates the rendering

struct SWorkArea
{
   SWorkArea(SWorkArea const&) =delete;
   SWorkArea( SWorkArea && ) = delete;
   explicit SWorkArea( uint32_t const minx, uint32_t const miny, uint32_t const maxx, uint32_t const maxy )
      : mMinX( minx ), mMinY( miny ), mMaxX( maxx ), mMaxY( maxy )
   { 
   }

   uint32_t mMinX;
   uint32_t mMinY;
   uint32_t mMaxX;
   uint32_t mMaxY;

   std::atomic<bool> mJobDone{ false };
};

class CRenderer
{
public:

   explicit CRenderer()
   {
      uint32_t const numProcessors = std::thread::hardware_concurrency();

      printf( "starting up %d job threads\n", numProcessors );

      for (uint32_t i = 0; i < numProcessors; ++i)
      {
         // worker threads
         mThreads.push_back( std::unique_ptr< std::thread >( new std::thread( [&]()
         {
            while (mShutdown == false)
            {
               SWorkArea * pWorkArea = NextWorkArea();

               if (pWorkArea!=nullptr)
               {
                  SWorkArea& workArea = *pWorkArea;
                  
                  // do work
                  int32_t const stepSize = skInitialStepSize;

                  for (uint32_t y = workArea.mMinY; y < workArea.mMaxY; y += stepSize)
                  {
                     for (uint32_t x = workArea.mMinX; x < workArea.mMaxX; x += stepSize)
                     {
                        CColor4f const color = mScene.DoIntersection( x, y );

                        for (uint32_t i = 0; i < stepSize; ++i)
                        {
                           for (uint32_t j = 0; j < stepSize; ++j)
                           {
                              SetPixelColor( x + i, y + j, color );
                           }
                        }
                     }
                  }

                  workArea.mJobDone = true;
               }
               else
               {
                  // no more jobs, just go to sleep
                  if (!mShutdown)
                  {
                     std::unique_lock<std::mutex> lock( mSleepControlMutex );
                     mSleepControl.wait( lock );
                  }
               }
            }
         } ) ) );
      }
   }

   ~CRenderer()
   {
      mShutdown = true;
      mSleepControl.notify_all();

      for (auto& threadObj : mThreads)
      {
         if (threadObj->joinable())
         {
            threadObj->join();
         }
      }
   }


   void Update( real32 const deltaTime )
   {
      if (IsDone())
      {
         mTime += deltaTime;
         mScene.Reset();
         NScene::BuildScene( mScene, mTime );
         mScene.SetSceneSize( mBufferWidth, mBufferHeight );
      }
   }

   bool IsDone() const
   {
      for (std::shared_ptr<SWorkArea> const& workArea : mWorkAreas)
      {
         if (!workArea->mJobDone)
         {
            return false;
         }
      }
      return true;
   }

   void Cancel()
   {
      // just declare remaining jobs done
      {
         std::lock_guard<std::mutex> lock( mJobMutex );
         while (mCurrentWorkArea < mWorkAreas.size())
         {
            mWorkAreas[mCurrentWorkArea++]->mJobDone = true;;
         }
      }

      // wait for all jobs to finish
      while (!IsDone())
      {
         std::this_thread::yield();
      }
   }

   void ResizeBuffer( uint32_t const width, uint32_t const height )
   {
      if (!IsDone())
      {
         Cancel();
      }

      if ( width != mBufferWidth || height != mBufferHeight )
      {
         mBuffer = std::unique_ptr< CColor4f >( reinterpret_cast<CColor4f*>(new uint8_t[sizeof( CColor4f ) * width * height]) );
         mBufferWidth = width;
         mBufferHeight = height;

         for (uint32_t x = 0; x < width; x++ )
         {
            for (uint32_t y = 0; y < height; y++ )
            {
               SetPixelColor( x, y, CColor4f( 0.5f, 0.6f, 0.7f ) );
            }
         }
      }
      mScene.SetSceneSize( width, height );
   }

   void RenderScene()
   {
      if (IsDone())
      {
         // create a whole bunch of render work areas
         {
            std::lock_guard<std::mutex> lock( mJobMutex );

            mWorkAreas.clear();
            mCurrentWorkArea = 0;

#if 1
            uint32_t const jobCount = std::thread::hardware_concurrency() * skJobCoreMultiplier;

            // break everything up into workable squares
            uint32_t const edgeJobCount = static_cast<uint32_t>(NMath::max_val( sqrtf( static_cast<real32>(jobCount) ), 1.f ));
            
            uint32_t const hStepSize = NMath::max_val( 1u, mBufferWidth / edgeJobCount );
            uint32_t const vStepSize = NMath::max_val( 1u, mBufferHeight / edgeJobCount );

            for (uint32_t y = 0; y < mBufferHeight; y += vStepSize)
            {
               for (uint32_t x = 0; x < mBufferWidth; x += hStepSize)
               {
                  mWorkAreas.push_back( std::make_shared< SWorkArea >( x, y, NMath::min_val( mBufferWidth, x + hStepSize ), NMath::min_val( mBufferHeight, y + vStepSize ) ) );
               }
            }
#else
            mWorkAreas.push_back( std::make_shared< SWorkArea >( mBufferWidth/2-2, mBufferHeight/2, mBufferWidth/2+2, mBufferHeight ) ) ;
#endif
         }
         mSleepControl.notify_all();
      }
   }

   CColor4f const* const GetBuffer() const
   {
      return mBuffer.get();
   }

   uint32_t GetBufferWidth() const
   {
      return mBufferWidth;
   }

   uint32_t GetBufferHeight() const
   {
      return mBufferHeight;
   }

   void SetPixelColor( uint32_t const x, uint32_t const y, CColor4f const & color )
   {
      if (x < mBufferWidth && y < mBufferHeight)
      {
         mBuffer.get()[y * mBufferWidth + x] = color;
      }
   }

   SWorkArea * NextWorkArea()
   {
      std::lock_guard<std::mutex> lock( mJobMutex );

      if ( !mWorkAreas.empty() )
      {
         if (mCurrentWorkArea < mWorkAreas.size())
         {
            return mWorkAreas[mCurrentWorkArea++].get();
         }
      }

      return nullptr;
   }

private:

   uint32_t mBufferWidth{ 0 };
   uint32_t mBufferHeight{ 0 };
   real32 mTime{ 0.f };
   std::unique_ptr< CColor4f > mBuffer;
   CRenderScene mScene;

   // thread control
   std::mutex mJobMutex;
   std::vector< std::shared_ptr< SWorkArea > > mWorkAreas;
   std::vector< std::unique_ptr< std::thread > > mThreads;

   std::atomic<uint32_t> mCurrentWorkArea;
   std::atomic<bool> mShutdown = false;
   std::condition_variable mSleepControl;
   std::mutex mSleepControlMutex;
};

//-----------------------------------------------------------------------------
// All of the windows handling stuff

namespace
{
   uint8_t* gBitmapBuffer = nullptr;
   HBITMAP ghBitmap = 0;

   TCHAR const * const skClassName = _T("RayMarcher");
   TCHAR const * const skWindowTitle = _T("RayMarcher");

   void fatal_exit( char const* const message )
   {
      TCHAR szBuf[80];
      LPVOID lpMsgBuf;
      DWORD dw = GetLastError();

      FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                     NULL,
                     dw,
                     MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
                     (LPTSTR)&lpMsgBuf,
                     0,
                     NULL );

      wsprintf( szBuf, _T("%s failed with error %d: %s"), message, dw, lpMsgBuf );

      MessageBox( NULL, szBuf, _T("Error"), MB_OK );

      LocalFree( lpMsgBuf );
      ExitProcess( dw );
   }

   LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
   {
      CRenderer * const pRenderer = reinterpret_cast<CRenderer*>(::GetWindowLongPtr( hWnd, GWLP_USERDATA ));

      switch (message)
      {
      case WM_CREATE:
         ::SetWindowLongPtr( hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(new CRenderer) );
         ::SetTimer( hWnd, 1, skTimerMilliseconds, NULL );
         break;
      case WM_DESTROY:
         // quit the entire application
         PostQuitMessage( 0 );
         pRenderer->Cancel();
         delete pRenderer;
         SetWindowLongPtr( hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(nullptr) );
         if (ghBitmap != 0)
         {
            ::DeleteObject( ghBitmap );
            ghBitmap = 0;
            gBitmapBuffer = nullptr;
         }
         break;
      case WM_TIMER:
         // force a repaint
         ::InvalidateRect( hWnd, NULL, FALSE ); 
         break;
      case WM_SHOWWINDOW:
         {
            // set the initial window size
            int32_t const winWidth = skDefaultWidth;
            int32_t const winHeight = skDefaultHeight;
            RECT rt;

            ::SetWindowPos(hWnd, NULL, 0, 0, winWidth, winHeight, SWP_NOZORDER | SWP_NOMOVE);

            // after setting the window size the client area will be too small, adjust 
            // the size of the window to take into account the window border
            ::GetClientRect(hWnd, &rt);
            int32_t const dwidth = winWidth - (rt.right - rt.left);
            int32_t const dheight = winHeight - (rt.bottom - rt.top);

            ::SetWindowPos(hWnd, NULL, 0, 0, winWidth + dwidth, winHeight + dheight, SWP_NOZORDER | SWP_NOMOVE);
         }
         break;
      case WM_SIZE:
         if (pRenderer)
         {
            pRenderer->Cancel();

            while (!pRenderer->IsDone())
            {
               std::this_thread::yield();
            }

            uint32_t const bufferWidth = static_cast<uint32_t>( (LOWORD( lParam ) + 3) & ~3 );
            uint32_t const bufferHeight = static_cast<uint32_t>( HIWORD( lParam ) );

            if (bufferWidth != 0 && bufferHeight != 0)
            {
               pRenderer->ResizeBuffer( bufferWidth, bufferHeight );

               // create a windows compatible bitmap buffer
               BITMAPINFOHEADER bitMapInfo =
               {
                  sizeof( BITMAPINFOHEADER ),
                  static_cast<int32_t>(pRenderer->GetBufferWidth()),
                  -static_cast<int32_t>(pRenderer->GetBufferHeight()),
                  1,
                  32,
                  BI_RGB,
                  0,
                  0,
                  0,
                  0,
                  0
               };

               if (ghBitmap != 0)
               {
                  ::DeleteObject( ghBitmap );
               }

               HDC const buffermemdc = ::CreateCompatibleDC( nullptr );
               ghBitmap = ::CreateDIBSection( buffermemdc,
                                             reinterpret_cast<BITMAPINFO const*>(&bitMapInfo),
                                             DIB_RGB_COLORS,
                                             reinterpret_cast<void**>(&gBitmapBuffer),
                                             nullptr,
                                             0 );
               ::DeleteDC( buffermemdc );
            }
         }
         break;
      case WM_PAINT:
         if ( pRenderer && gBitmapBuffer != nullptr )
         {
            PAINTSTRUCT ps;
            HDC const hdc = ::BeginPaint( hWnd, &ps );

            HDC const buffermemdc = ::CreateCompatibleDC( nullptr );
            
            // do conversion
#if !SHOW_RENDER_PROGRESS()
            if (pRenderer->IsDone())
#endif
            {
               for (uint32_t y = 0; y < pRenderer->GetBufferHeight(); ++y)
               {
                  uint32_t const voffset = y * pRenderer->GetBufferWidth();
                  for (uint32_t x = 0; x < pRenderer->GetBufferWidth(); ++x)
                  {
                     CColor4f const& color = pRenderer->GetBuffer()[voffset + x];

                     uint32_t const bitmapPosition = (voffset + x) * 4;
                     gBitmapBuffer[bitmapPosition + 0] = static_cast<uint8_t>(NMath::min_val( 1.f, color.GetBlue() ) * 255.0);
                     gBitmapBuffer[bitmapPosition + 1] = static_cast<uint8_t>(NMath::min_val( 1.f, color.GetGreen() ) * 255.0);
                     gBitmapBuffer[bitmapPosition + 2] = static_cast<uint8_t>(NMath::min_val( 1.f, color.GetRed() ) * 255.0);
                  }
               }
            }

            // present
            ::SelectObject( buffermemdc, ghBitmap );
            ::BitBlt( hdc, 0, 0, static_cast<int>(pRenderer->GetBufferWidth()), static_cast<int>(pRenderer->GetBufferHeight()), buffermemdc, 0, 0, SRCCOPY );



#if 0
            {
               RECT rt;
               ::GetClientRect( hWnd, &rt );
               int32_t const dwidth = (rt.right - rt.left);
               int32_t const dheight = (rt.bottom - rt.top);
               draw_field( hdc, dwidth, dheight );
            }
#endif

            // clean up
            ::DeleteDC( buffermemdc );

            ::EndPaint( hWnd, &ps );

            // repaint is done, request a new scene
            if (pRenderer->IsDone())
            {
               pRenderer->Update( 0.1f );
               pRenderer->RenderScene();
            }

         }
         break;

      case WM_KEYDOWN:
         if (wParam == VK_ESCAPE)
         {
            ::DestroyWindow( hWnd );
         }
         break;
      }

      return DefWindowProc( hWnd, message, wParam, lParam );
   }

   // ---------------------

   void register_class( HINSTANCE hInstance )
   {
      WNDCLASSEX wcex;

      ::ZeroMemory( &wcex, sizeof( WNDCLASSEX ) );

      wcex.cbSize = sizeof( WNDCLASSEX );

      wcex.style = CS_HREDRAW | CS_VREDRAW;
      wcex.lpfnWndProc = reinterpret_cast<WNDPROC>(WndProc);
      wcex.cbClsExtra = 0;
      wcex.cbWndExtra = 0;
      wcex.hInstance = hInstance;
      wcex.hIcon = ::LoadIcon( nullptr, IDI_APPLICATION );
      wcex.hCursor = ::LoadCursor( nullptr, IDC_ARROW );
      wcex.hbrBackground = NULL;
      wcex.lpszMenuName = nullptr;
      wcex.lpszClassName = skClassName;

      if (::RegisterClassEx( &wcex ) == 0 )
      {
         fatal_exit( "register_class()" );
      }
   }

   // ---------------------

   bool init_instance( HINSTANCE hInstance, int const cmdShow )
   {
      DWORD const style =
#if CAN_BE_RESIZED()
         WS_OVERLAPPEDWINDOW
#else
         (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX)
#endif
         ;
      HWND const hWnd = ::CreateWindow( skClassName,
                                        skWindowTitle,
                                        style,
                                        CW_USEDEFAULT,
                                        0,
                                        CW_USEDEFAULT,
                                        0,
                                        NULL,
                                        NULL,
                                        hInstance,
                                        NULL );

      if (!hWnd)
      {
         fatal_exit( "init_instance()" );
      }
      else
      {
         ShowWindow( hWnd, cmdShow );
         UpdateWindow( hWnd );
      }

      return TRUE;
   }

   // ---------------------

   BOOL CALLBACK static_console_handler( DWORD dwCtrlType )
   {
      (dwCtrlType);
      ::PostQuitMessage( 0 );
      return FALSE;
   }
}

int APIENTRY WinMain(
   _In_ HINSTANCE hInstance,
   _In_opt_ HINSTANCE hPrevInstance,
   _In_ LPSTR     lpCmdLine,
   _In_ int       nCmdShow)
{
   (hPrevInstance);
   (lpCmdLine);

#if ENABLE_CONSOLE()
   ::AllocConsole();

   // make sure there is a graceful shutdown
   ::SetConsoleCtrlHandler(static_console_handler, true);

   // want anything going to stdout or stderr to
   // go into the console
   FILE* oldStdOut = nullptr;
   FILE* oldStdErr = nullptr;
   freopen_s(&oldStdOut, "CONOUT$", "wt", stdout);
   freopen_s(&oldStdErr, "CONOUT$", "wt", stderr);
#endif

   register_class(hInstance);

   init_instance(hInstance, nCmdShow);

   MSG msg;

   // Main message loop:
   while (GetMessage(&msg, NULL, 0, 0))
   {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
   }

   return 0;
}
