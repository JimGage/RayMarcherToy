#pragma once
#ifndef MATHCLASSES_H
#define MATHCLASSES_H

#include <cstdint>
#include <math.h>
#include <algorithm>
#include <intrin.h>

typedef float real32;

#define USE_INTRINSICS() 1

namespace NMath
{
   real32 const gkPi32 = 3.14159265358979323846264338327950288f;
   real32 const gkMaxFloat = 3.402823466e+38f;
   real32 const gkMinFloat = 1.175494351e-38f;
   real32 const gkEpsilon = 1.192092896e-7f;

   template<class T>
   inline T min_val(T const lhs, T const rhs) { return lhs < rhs ? lhs : rhs; }
   template<class T>
   inline T max_val(T const lhs, T const rhs) { return lhs > rhs ? lhs : rhs; }
   template<class T>
   inline T clamp( T const lhs, T const value, T const rhs ) { return lhs < value ? lhs : (rhs > value ? rhs : value);  }

   inline real32 random_value(real32 const lower, real32 const upper)
   {
      return (rand() / real32(RAND_MAX)) * (upper - lower) + lower;
   }

   inline real32 lerp( real32 const a, real32 const b, real32 const t )
   {
      return a + ((b - a) * t);
   }

   inline real32 AbsF(real32 const value)
   {
      return ::fabsf(value);
   }

   inline real32 Sign(real32 const value) { return value >= 0.f ? 1.f : -1.f; }
   inline real32 FastFSel(real32 const compare, real32 const minVal, real32 maxVal) { return compare >= 0.f ? maxVal : minVal; }

   inline bool small_enough(real32 const value, real32 const epsilon = gkEpsilon * 10.f)
   {
      return AbsF(value) < epsilon;
   }

   inline bool close_enough(real32 const lhs, real32 const rhs, real32 const epsilon = gkEpsilon * 10.f)
   {
      real32 const delta = AbsF(lhs - rhs);
      real32 const scaledEpsilon = max_val(AbsF(lhs), AbsF(rhs)) * epsilon;
      return delta < scaledEpsilon;
   }

   inline uint32_t const NextPowerOfTwo(uint32_t const value)
   {
      for (uint32_t i = 0; i < 32; ++i)
      {
         uint32_t const power_of_two = 1u << i;
         if (power_of_two >= value)
         {
            return power_of_two;
         }
      }
      return 1u << 31;
   }
}

//-------------------------------------------------------------------------

class CVector3f
{
public:
   CVector3f(real32 const x, real32 const y, real32 const z)
      : mX(x), mY(y), mZ(z) { }
   CVector3f const operator-() const { return CVector3f(-mX, -mY, -mZ); }
   CVector3f const operator+(CVector3f const& rhs) const { return CVector3f(mX + rhs.mX, mY + rhs.mY, mZ + rhs.mZ); }
   CVector3f const operator-(CVector3f const& rhs) const { return CVector3f(mX - rhs.mX, mY - rhs.mY, mZ - rhs.mZ); }
   CVector3f const operator*(CVector3f const& rhs) const { return CVector3f(mX * rhs.mX, mY * rhs.mY, mZ * rhs.mZ); }
   CVector3f const operator*(real32 const rhs) const { return CVector3f(mX * rhs, mY * rhs, mZ * rhs); }
   CVector3f const operator/(real32 const rhs) const { return CVector3f(mX / rhs, mY / rhs, mZ / rhs); }
   CVector3f const& operator+=(CVector3f const& rhs) { mX += rhs.mX; mY += rhs.mY; mZ += rhs.mZ; return *this; }
   CVector3f const& operator-=(CVector3f const& rhs) { mX -= rhs.mX; mY -= rhs.mY; mZ -= rhs.mZ; return *this; }
   bool operator==(CVector3f const& rhs) const { return mX == rhs.mX && mY == rhs.mY && mZ == rhs.mZ; }
   bool operator!=(CVector3f const& rhs) const { return mX != rhs.mX || mY != rhs.mY || mZ != rhs.mZ; }

#if USE_INTRINSICS()
   real32 const DotProductWith( CVector3f const& rhs ) const
   {
      // load the values
      __m128 const value0 = _mm_loadu_ps( &mX );
      __m128 const value1 = _mm_loadu_ps( &rhs.mX );

      uint32_t const skUpperMask = 0x70; // calculate off of the first three floats
      uint32_t const skLowerMask = 0x01; // the result goes into r0 

      // dot product
      __m128 const dotProduct = _mm_dp_ps( value0, value1, skUpperMask | skLowerMask );

      // get the results
      return _mm_cvtss_f32( dotProduct );
   }
#else
   real32 const DotProductWith(CVector3f const& rhs) const { return mX * rhs.mX + mY * rhs.mY + mZ * rhs.mZ; }
#endif

   CVector3f const CrossProductWith(CVector3f const& rhs) const
   {
      return CVector3f(mY * rhs.mZ - mZ * rhs.mY, mZ * rhs.mX - mX * rhs.mZ, mX * rhs.mY - mY * rhs.mX);
   }
   static real32 const Dot(CVector3f const& lhs, CVector3f const& rhs)
   {
      return lhs.DotProductWith(rhs);
   }
   static CVector3f Cross(CVector3f const& lhs, CVector3f const& rhs)
   {
      return lhs.CrossProductWith(rhs);
   }
   real32 const MagnitudeSquared() const { return DotProductWith(*this); }
   real32 const Magnitude() const { 
#if USE_INTRINSICS()
      __m128 const value = _mm_loadu_ps( &mX );

      // dot product
      uint32_t constexpr skUpperMask = 0x70; // calculate off of the first three floats
      uint32_t constexpr skLowerMask = 0x01; // the result goes into r0 

      __m128 const dotProduct = _mm_dp_ps( value, value, skUpperMask | skLowerMask );

      // calculate the square root
      return  _mm_cvtss_f32( _mm_sqrt_ss( dotProduct ) );
#else
      return sqrtf(MagnitudeSquared()); 
#endif
   }
   CVector3f const AsNormalized() const 
   { 
#if USE_INTRINSICS()
      __m128 const value = _mm_loadu_ps(&mX);

      // dot product
      uint32_t const skUpperMask = 0x70; // calculate off of the first three floats
      uint32_t const skLowerMask = 0x01; // the result goes into r0 

      __m128 const dotProduct = _mm_dp_ps(value, value, skUpperMask | skLowerMask);

      // approximate reciprocal square root
      real32 inverseMagnitude = _mm_cvtss_f32(_mm_rsqrt_ss(dotProduct));

      real32 const magSquared = _mm_cvtss_f32(dotProduct);

      // Netwon-Raphson refinement
      real32 const halfVal = 0.5f * magSquared;
      inverseMagnitude = inverseMagnitude * (1.5f - inverseMagnitude * inverseMagnitude * halfVal);

      return CVector3f(mX * inverseMagnitude, mY * inverseMagnitude, mZ * inverseMagnitude);
#else
      return *this / Magnitude(); 
#endif
   }
   real32 const GetX() const { return mX; }
   real32 const GetY() const { return mY; }
   real32 const GetZ() const { return mZ; }
   real32 const operator[](int const index) const { return (&mX)[index]; }
   real32& operator[](int const index) { return (&mX)[index]; }
   static CVector3f const FromBaryCentric(CVector3f const& vert0, CVector3f const& vert1, CVector3f const& vert2, real32 const u, real32 const v)
   {
      return (vert1 - vert0) * u + (vert2 - vert0) * v + vert0;
   }
   static CVector3f const Zero() { return CVector3f(0.f, 0.f, 0.f); }
   static CVector3f const One() { return CVector3f( 1.f, 1.f, 1.f ); }
   static CVector3f const Up() { return CVector3f(0.f, 1.f, 0.f); }

public:
   union 
   {
      real32 mX;
      real32 x;
   };
   union
   {
      real32 mY;
      real32 y;
   };
   union
   {
      real32 mZ;
      real32 z;
   };
};

//-------------------------------------------------------------------------

class CRelAngle
{
public:
   static CRelAngle FromRadians(real32 const radians) { return CRelAngle(radians); }
   static CRelAngle FromDegrees(real32 const degrees) { return CRelAngle(degrees * NMath::gkPi32 / 180.f); }
   static CRelAngle NoRotation() { return CRelAngle(0.f); }

   real32 AsRadians() const { return mRadians; }
   real32 AsDegrees() const { return mRadians / NMath::gkPi32 * 180.f; }

   CRelAngle operator-() const { return CRelAngle(-mRadians); }
   CRelAngle operator+(CRelAngle const rhs) const { return CRelAngle(mRadians + rhs.mRadians); }
   CRelAngle operator-(CRelAngle const rhs) const { return CRelAngle(mRadians - rhs.mRadians); }
   CRelAngle operator*(real32 const rhs) const { return CRelAngle(mRadians * rhs); }
   CRelAngle operator/(real32 const rhs) const { return CRelAngle(mRadians / rhs); }
   CRelAngle const& operator+=(CRelAngle const rhs);
   CRelAngle const& operator-=(CRelAngle const rhs);
   CRelAngle const& operator*=(real32 const rhs);
   CRelAngle const& operator/=(real32 const rhs);

private:
   explicit CRelAngle(real32 const radians)
      : mRadians(radians)
   {
   }
   real32 mRadians;
};


inline CRelAngle const& CRelAngle::operator+=(CRelAngle const rhs)
{
   mRadians += rhs.mRadians;
   return *this;
}

inline CRelAngle const& CRelAngle::operator-=(CRelAngle const rhs)
{
   mRadians -= rhs.mRadians;
   return *this;
}

inline CRelAngle const& CRelAngle::operator*=(real32 const rhs)
{
   mRadians *= rhs;
   return *this;
}

inline CRelAngle const& CRelAngle::operator/=(real32 const rhs)
{
   mRadians /= rhs;
   return *this;
}

//-------------------------------------------------------------------------

class CTransform4f
{
public:
   explicit CTransform4f(real32 const a00, real32 const a01, real32 const a02, real32 const a03,
      real32 const a10, real32 const a11, real32 const a12, real32 const a13,
      real32 const a20, real32 const a21, real32 const a22, real32 const a23)
      : m00(a00), m01(a01), m02(a02), m03(a03)
      , m10(a10), m11(a11), m12(a12), m13(a13)
      , m20(a20), m21(a21), m22(a22), m23(a23) { }

   static CTransform4f const Identity()
   {
      return CTransform4f(
         1.f, 0.f, 0.f, 0.f,
         0.f, 1.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f);
   }

   static CTransform4f const FromVectors(CVector3f const & x, CVector3f const & y, CVector3f const &z, CVector3f const & t)
   {
      return CTransform4f(
         x.GetX(), y.GetX(), z.GetX(), t.GetX(),
         x.GetY(), y.GetY(), z.GetY(), t.GetY(),
         x.GetZ(), y.GetZ(), z.GetZ(), t.GetZ());
   }

   static CTransform4f const FromLeftForwardUp(CVector3f const & left, CVector3f const & forward, CVector3f const & up, CVector3f const & translation)
   {
      return FromVectors(-left, up, -forward, translation);
   }

   static CTransform4f const FromRightForwardUp(CVector3f const & right, CVector3f const & forward, CVector3f const & up, CVector3f const & translation)
   {
      return FromVectors(right, up, -forward, translation);
   }

   static CTransform4f const CrossProductForm(CVector3f const & vector)
   {
      return CTransform4f(
                    0.f, -vector.GetZ(),  vector.GetY(), 0.f,
          vector.GetZ(),            0.f, -vector.GetX(), 0.f,
         -vector.GetY(),  vector.GetX(),            0.f, 0.f );
   }

      // static functions

   static CTransform4f const Translate(real32 const x, real32 const y, real32 const z)
   {
      return CTransform4f(1.f, 0.f, 0.f, x,
                          0.f, 1.f, 0.f, y,
                          0.f, 0.f, 1.f, z);

   }

   static CTransform4f const Translate(CVector3f const & translation)
   {
      return Translate(translation.GetX(), translation.GetY(), translation.GetZ());
   }

   static CTransform4f const Scale(real32 const x, real32 const y, real32 const z)
   {
      return CTransform4f(x, 0.f, 0.f, 0.f,
                          0.f, y, 0.f, 0.f,
                          0.f, 0.f, z, 0.f);
   }

   static CTransform4f const Scale(CVector3f const & scale)
   {
      return Scale(scale.GetX(), scale.GetY(), scale.GetZ());
   }

   static CTransform4f const RotateXRadians(real32 const radians)
   {
      real32 const c = ::cosf(radians);
      real32 const s = ::sinf(radians);
      return CTransform4f(1.f, 0.f, 0.f, 0.f,
                          0.f, c, -s, 0.f,
                          0.f, s, c, 0.f);
   }

   static CTransform4f const RotateYRadians(real32 const radians)
   {
      real32 const c = ::cosf(radians);
      real32 const s = ::sinf(radians);
      return CTransform4f(c, 0.f, s, 0.f,
                          0.f, 1.f, 0.f, 0.f,
                          -s, 0.f, c, 0.f);
   }
   static CTransform4f const RotateZRadians(real32 const radians)
   {
      real32 const c = ::cosf(radians);
      real32 const s = ::sinf(radians);
      return CTransform4f(c, -s, 0.f, 0.f,
                          s, c, 0.f, 0.f,
                          0.f, 0.f, 1.f, 0.f);
   }

   static CTransform4f const RotateX(CRelAngle const & angle)
   {
      real32 const c = ::cosf(angle.AsRadians());
      real32 const s = ::sinf(angle.AsRadians());
      return CTransform4f(1.f, 0.f, 0.f, 0.f,
                          0.f, c, -s, 0.f,
                          0.f, s, c, 0.f);
   }

   static CTransform4f const RotateY(CRelAngle const & angle)
   {
      real32 const c = ::cosf(angle.AsRadians());
      real32 const s = ::sinf(angle.AsRadians());
      return CTransform4f(c, 0.f, s, 0.f,
                          0.f, 1.f, 0.f, 0.f,
                          -s, 0.f, c, 0.f);
   }
   
   static CTransform4f const RotateZ(CRelAngle const & angle)
   {
      real32 const c = ::cosf(angle.AsRadians());
      real32 const s = ::sinf(angle.AsRadians());
      return CTransform4f(c, -s, 0.f, 0.f,
                          s, c, 0.f, 0.f,
                          0.f, 0.f, 1.f, 0.f);
   }

   // non-static functions

   CVector3f const GetForward() const
   {
      return CVector3f(-m02, -m12, -m22);
   }

   CVector3f const GetBackward() const
   {
      return CVector3f(m02, m12, m22);
   }

   CVector3f const GetRight() const
   {
      return CVector3f(m00, m10, m20);
   }

   CVector3f const GetLeft() const
   {
      return CVector3f(-m00, -m10, -m20);
   }

   CVector3f const GetUp() const
   {
      return CVector3f(m01, m11, m21);
   }

   CVector3f const GetDown() const
   {
      return CVector3f(-m01, -m11, -m21);
   }


   CTransform4f const operator*(CTransform4f const & rhs) const
   {
      return CTransform4f(
         m00 * rhs.m00 + m01 * rhs.m10 + m02 * rhs.m20,
         m00 * rhs.m01 + m01 * rhs.m11 + m02 * rhs.m21,
         m00 * rhs.m02 + m01 * rhs.m12 + m02 * rhs.m22,
         m00 * rhs.m03 + m01 * rhs.m13 + m02 * rhs.m23 + m03,

         m10 * rhs.m00 + m11 * rhs.m10 + m12 * rhs.m20,
         m10 * rhs.m01 + m11 * rhs.m11 + m12 * rhs.m21,
         m10 * rhs.m02 + m11 * rhs.m12 + m12 * rhs.m22,
         m10 * rhs.m03 + m11 * rhs.m13 + m12 * rhs.m23 + m13,

         m20 * rhs.m00 + m21 * rhs.m10 + m22 * rhs.m20,
         m20 * rhs.m01 + m21 * rhs.m11 + m22 * rhs.m21,
         m20 * rhs.m02 + m21 * rhs.m12 + m22 * rhs.m22,
         m20 * rhs.m03 + m21 * rhs.m13 + m22 * rhs.m23 + m23);
   }
   CVector3f const operator*(CVector3f const & rhs) const
   {
#if USE_INTRINSICS()

      __m128 const vector = _mm_set_ps(1.f, rhs.GetZ(), rhs.GetY(), rhs.GetX());

      __m128 const row0 = _mm_loadu_ps(&m00);
      __m128 const row1 = _mm_loadu_ps(&m10);
      __m128 const row2 = _mm_loadu_ps(&m20);


      uint32_t const skUpperMask = 0xF0;
      uint32_t const skLowerMask = 0x01;

      __m128 resultX = _mm_dp_ps(vector, row0, skUpperMask | skLowerMask);
      __m128 resultY = _mm_dp_ps(vector, row1, skUpperMask | skLowerMask);
      __m128 resultZ = _mm_dp_ps(vector, row2, skUpperMask | skLowerMask);

      return CVector3f(_mm_cvtss_f32(resultX), _mm_cvtss_f32(resultY), _mm_cvtss_f32(resultZ));
#else
      return CVector3f(
         m00 * rhs.GetX() + m01 * rhs.GetY() + m02 * rhs.GetZ() + m03,
         m10 * rhs.GetX() + m11 * rhs.GetY() + m12 * rhs.GetZ() + m13,
         m20 * rhs.GetX() + m21 * rhs.GetY() + m22 * rhs.GetZ() + m23);
#endif
   }

   CVector3f const GetXBasis() const { return CVector3f(m00, m10, m20); }
   CVector3f const GetYBasis() const { return CVector3f(m01, m11, m21); }
   CVector3f const GetZBasis() const { return CVector3f(m02, m12, m22); }
   CVector3f const GetTranslation() const { return CVector3f(m03, m13, m23); }
   
   CVector3f const GetColumn(int32_t const columnIndex) const
   {
      return CVector3f(*(&m00 + columnIndex), *(&m10 + columnIndex), *(&m20 + columnIndex));
   }

   CVector3f const Rotate(CVector3f const & rhs) const
   {
      return CVector3f(
         m00 * rhs.GetX() + m01 * rhs.GetY() + m02 * rhs.GetZ(),
         m10 * rhs.GetX() + m11 * rhs.GetY() + m12 * rhs.GetZ(),
         m20 * rhs.GetX() + m21 * rhs.GetY() + m22 * rhs.GetZ());
   }

   CVector3f const TransposeRotate(CVector3f const & rhs) const
   {
      return CVector3f(
         m00 * rhs.GetX() + m10 * rhs.GetY() + m20 * rhs.GetZ(),
         m01 * rhs.GetX() + m11 * rhs.GetY() + m21 * rhs.GetZ(),
         m02 * rhs.GetX() + m12 * rhs.GetY() + m22 * rhs.GetZ());
   }

   CVector3f const TransposeMultiply(CVector3f const & rhs) const
   {
      return TransposeRotate(CVector3f(rhs.GetX() - m03, rhs.GetY() - m13, rhs.GetZ() - m23));
   }

   CTransform4f const Transpose() const
   {
      return CTransform4f(
         m00, m10, m20, 0.f,
         m01, m11, m21, 0.f,
         m02, m12, m22, 0.f);
   }

   real32 const GetDeterminant() const
   {
      // This is greatly reduced because the last row is always 0 0 0 1
      return m00*(m11*m22 - m12*m21) + m01*(m12*m20 - m10*m22) + m02*(m10*m21 - m11*m20);
   }

   CTransform4f const GetInverse() const
   {
      real32 const determinant = GetDeterminant();

      if (fabsf(determinant) < 0.00001f)
      {
         return CTransform4f::Identity();
      }

      real32 const inverseDet = 1.f / determinant;

      real32 const t03 = -((m12*m23 - m13*m22)*m01 - (m02*m23 - m03*m22)*m11 + (m02*m13 - m03*m12)*m21);
      real32 const t13 = ((m12*m23 - m13*m22)*m00 - (m02*m23 - m03*m22)*m10 + (m02*m13 - m03*m12)*m20);
      real32 const t23 = -((m10*m21 - m11*m20)*m03 - (m00*m21 - m01*m20)*m13 + (m00*m11 - m01*m10)*m23);

      return CTransform4f(
         (m11*m22 - m12*m21)*inverseDet, (m02*m21 - m01*m22)*inverseDet, (m01*m12 - m02*m11)*inverseDet, t03*inverseDet,
         (m12*m20 - m10*m22)*inverseDet, (m00*m22 - m02*m20)*inverseDet, (m02*m10 - m00*m12)*inverseDet, t13*inverseDet,
         (m10*m21 - m11*m20)*inverseDet, (m01*m20 - m00*m21)*inverseDet, (m00*m11 - m01*m10)*inverseDet, t23*inverseDet);
   }

   CTransform4f AsOrthonormalized() const
   {
      CVector3f const column0 = GetColumn(0).AsNormalized();

      CVector3f const column2 = CVector3f::Cross(column0, GetColumn(1)).AsNormalized();

      CVector3f const column1 = CVector3f::Cross(column2, column0);

      return CTransform4f::FromVectors(column0, column1, column2, GetTranslation());
   }

   real32 m00, m01, m02, m03;
   real32 m10, m11, m12, m13;
   real32 m20, m21, m22, m23;
};

//-------------------------------------------------------------------------

class CColor4f
{
public:
   explicit CColor4f(real32 const red, real32 const green, real32 const blue)
      : mRed(red), mGreen(green), mBlue(blue) { }
   explicit CColor4f(CVector3f const& vectorColor)
      : mRed(vectorColor.GetX()), mGreen(vectorColor.GetY()), mBlue(vectorColor.GetZ()) { }
   explicit CColor4f(uint32_t const colorCode )
      : mRed(((colorCode >> 16) & 0xff)/255.f), mGreen(((colorCode >> 8) & 0xff)/255.f), mBlue(((colorCode >> 0) & 0xff)/255.f) { }

   real32 const GetRed() const { return mRed; }
   real32 const GetGreen() const { return mGreen; }
   real32 const GetBlue() const { return mBlue; }

   CColor4f const operator-() const { return CColor4f(-mRed, -mGreen, -mBlue); }
   CColor4f const operator+(CColor4f const& rhs) const { return CColor4f(mRed + rhs.mRed, mGreen + rhs.mGreen, mBlue + rhs.mBlue); }
   CColor4f const operator-(CColor4f const& rhs) const { return CColor4f(mRed - rhs.mRed, mGreen - rhs.mGreen, mBlue - rhs.mBlue); }
   CColor4f const operator*(CColor4f const& rhs) const { return CColor4f(mRed * rhs.mRed, mGreen * rhs.mGreen, mBlue * rhs.mBlue); }
   CColor4f const operator/(CColor4f const& rhs) const { return CColor4f(mRed / rhs.mRed, mGreen / rhs.mGreen, mBlue / rhs.mBlue); }
   CColor4f const operator*(real32 const rhs) const { return CColor4f(mRed * rhs, mGreen * rhs, mBlue * rhs); }
   CColor4f const operator/(real32 const rhs) const { return CColor4f(mRed / rhs, mGreen / rhs, mBlue / rhs); }
   CColor4f const& operator+=(CColor4f const& rhs) { mRed += rhs.mRed; mGreen += rhs.mGreen; mBlue += rhs.mBlue; return *this; }
   CColor4f const& operator-=(CColor4f const& rhs) { mRed -= rhs.mRed; mGreen -= rhs.mGreen; mBlue -= rhs.mBlue; return *this; }
   CColor4f const& operator*=(CColor4f const& rhs) { mRed *= rhs.mRed; mGreen *= rhs.mGreen; mBlue *= rhs.mBlue; return *this; }
   CColor4f const& operator/=(CColor4f const& rhs) { mRed /= rhs.mRed; mGreen /= rhs.mGreen; mBlue /= rhs.mBlue; return *this; }
   CColor4f const& operator*=(real32 const rhs) { mRed *= rhs; mGreen *= rhs; mBlue *= rhs; return *this; }
   bool operator==(CColor4f const& rhs) const { return mRed == rhs.mRed && mGreen == rhs.mGreen && mBlue == rhs.mBlue; }
   bool operator!=(CColor4f const& rhs) const { return mRed != rhs.mRed || mGreen != rhs.mGreen || mBlue != rhs.mBlue; }

   static CColor4f const Lerp(CColor4f const& lhs, CColor4f const& rhs, real32 const t) { return (lhs * (1.f - t)) + (rhs * t); }
   static CColor4f const Black() { return CColor4f(0.f, 0.f, 0.f); }
   static CColor4f const White() { return CColor4f(1.f, 1.f, 1.f); }

private:
   real32 mRed;
   real32 mGreen;
   real32 mBlue;
};

#endif