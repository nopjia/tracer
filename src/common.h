#ifndef COMMON_H
#define COMMON_H

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned int uint;

// constants
#define EPS 0.001f

#define WINDOW_W    640
#define WINDOW_H    480
#define PIXSCALE    2
#define DELTA_T     0.1f
#define BLOCK_SIZE  32

// rendering constants
#define PATH_DEPTH  8

// options
#define TRI_NORM_INTERP



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