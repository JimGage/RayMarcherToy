//----------------------------------------------------------------------------------------
//
// I've made the scene function a separate file to make it a little bit easier
// to modify a scene. There's a lot of little convenience functions to make it easier to
// edit a scene.
//
// Use += to append an object to the scene
// Use << to assign values 
//
// scene << camera ;   set the camera
// scene += light ;    add a light to the scene
// scene += object;    add an object to the scene
//
// object << translate( vector ) * scale( vector ) * scale( scalar ) * rotate( vector ) * rotatex( scalar );
// object << color( 1.f, 1.f, 1.f );
// object << checker( blue, green );
// object << ( checker( blue, green ) << scale( 10.f ) * rotatex(45.f ) ) ;
//
// camera( center, lookat, [,fov] [, is_vertical_fov ] );
// pointlight( center, color );
// sphere( radius )
// sphere( center, radius )
// cube( size )
//
// custom( function ) // a custom object takes a lambda as a parameter
//
// This custom object creates a sphere with a radius of 3 at the position < 0, 4, 10 >:
// scene += custom( []( vector3 pos ) { return pos.Magnitude() - 3.f;  } ) << translate( 0.f, 4.f, 10.f );
//
//----------------------------------------------------------------------------------------

#define torus(minorRadius, majorRadius) custom( []( vector3 pos ) { return length( vector3( length(vector3( pos.x, 0.f, pos.z ) ) -  majorRadius, pos.y, 0.f ) ) - minorRadius;  } )

color const steel_blue( 0x4682b4 );
color const spring_green( 0x00ff7f );
color const slate_gray( 0x708090 );

auto rmod = []( real32 x, real32 y ) { return x - y * roundf( x / y ); };

// camera
scene << camera( vector3( 0.f, 15.f, 15.f ), vector3( 0.f, 0.f, 0.f ) );
//scene * rotatey( time * 20.f );

// setup some lights
scene += ambientlight( color( 0.1f, 0.1f, 0.1f ) );
scene += directionallight( vector3( 0.f, -1.f, 0.f ), color( 0.1f, 0.1f, 0.2f ) );

// choose between point and spot light
#if 1
scene += pointlight(vector3(0.f, 5.f + sinf(time * 3.f), 0.f), color(0.9f, 0.9f, 0.8f) * 10.f) << attenuation{ .linear = 0.7f, .exponential = 0.3f };
#else
scene += spotlight( vector3(0.f, 20.f, 0.f ), vector3(0.f, -1.f, 0.f ), 10.f, color( 1.f, 1.f, 1.f ) ) << attenuation{ .constant = 0.8f, .linear = 0.2f, .exponential = 0.f }
<< rotatez(sinf(time*3.f) * 10.f);
#endif


// some test objects
scene += plane( vector3( 0.f, 1.f, 0.f ) ) << translate( 0.f, -5.f, 0.f ) << checker( color(0xeeeeee) , color(0xaaaaaa) );


scene += csg_difference( { torus( 1.f,2.f ), cube( 4.f ) << translate( 2, 0, 2 ) } ) << translate(-6,0,0)  << surface{ .dielectric = 0.4f };

scene += csg_smoothunion(
   {
      cube( 3.f ) << translate( 1.25f,0,0 ) << color( 0x00aaaa) ,
      sphere( 1.5f ) << translate( -1.25f,0,0 ) << color( 0xaa1111) 
   },
   0.5f // blend factor
) << translate( 6, 0, 0 ) << surface{ .metallic = 0.4f };

scene += blend(
   {
      torus( 1.f,2.f ) << color( 0.1f,0.7f,0.1f ),
      cube( 3.f ),
      sphere( 3.f ) << color( 0.5f,0.1f,0.1f )
   }, 1.f + sinf( time * 3.f - (3.1415926f/2.f) ) ) << surface{ .dielectric = 0.3f };
