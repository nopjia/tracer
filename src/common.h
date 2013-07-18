#ifndef COMMON_H
#define COMMON_H

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned int uint;

//---------------------------------------------------------
// MATH CONSTANTS
//---------------------------------------------------------

#define M_E         2.71828182845904523536028747135266250f   /* e */
#define M_LOG2E     1.44269504088896340735992468100189214f   /* log 2e */
#define M_LOG10E    0.434294481903251827651128918916605082f  /* log 10e */
#define M_LN2       0.693147180559945309417232121458176568f  /* log e2 */
#define M_LN10      2.30258509299404568401799145468436421f   /* log e10 */
#define M_PI        3.14159265358979323846264338327950288f   /* pi */
#define M_PI_2      1.57079632679489661923132169163975144f   /* pi/2 */
#define M_PI_4      0.785398163397448309615660845819875721f  /* pi/4 */
#define M_1_PI      0.318309886183790671537767526745028724f  /* 1/pi */
#define M_2_PI      0.636619772367581343075535053490057448f  /* 2/pi */
#define M_2_SQRTPI  1.12837916709551257389615890312154517f   /* 2/sqrt(pi) */
#define M_SQRT2     1.41421356237309504880168872420969808f   /* sqrt(2) */
#define M_SQRT1_2   0.707106781186547524400844362104849039f  /* 1/sqrt(2) */


//---------------------------------------------------------
// PROGRAM CONSTANTS
//---------------------------------------------------------

#define EPS   0.001f

#define WINDOW_W    512
#define WINDOW_H    512
#define PIXSCALE    2
#define DELTA_T     0.01f
#define BLOCK_SIZE  32

// rendering constants
#define THFL_NONE       0x0000
#define THFL_PATH_RUN   0x0001
#define FOV 30
#define PATH_DEPTH 3

// options
#define TRI_NORM_INTERP
#define GAMMA_CORRECT
//#define FOCAL_BLUR


//---------------------------------------------------------
// OPENGL
//---------------------------------------------------------

// vertex attribute indexes
#define POSITION_ATTR                 0
#define NORMAL_ATTR                   1
#define UV_ATTR                       2

// uniform buffer binding points
#define PER_FRAME_UBO_BINDING         0
#define LIGHT_UBO_BINDING             1
#define MESH_MATERIAL_ARRAY_BINDING   2
#define POSITION_ARRAY_BINDING        3

// sampler binding points
#define UNUSED_TEXTURE                0
#define RENDER_TEXTURE                1

// max values
#define MAX_TEXTURE_ARRAYS            10
#define NUM_OBJECTS_MAX               500
#define NUM_MESHES_MAX                500
#define MAX_POINT_LIGHTS              8

#endif  // COMMON_H