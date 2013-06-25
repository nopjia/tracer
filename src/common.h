#ifndef COMMON_H
#define COMMON_H

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned int uint;

// constants
#define EPS 0.001f

const uint WINDOW_W = 640;
const uint WINDOW_H = 480;
const uint FOV = 45;
const uint BLOCK_SIZE = 32;
const float DELTA_T = 0.1f;

// Vertex attribute indexes
const uint POSITION_ATTR                    = 0;
const uint NORMAL_ATTR                      = 1;
const uint UV_ATTR                          = 2;

// Uniform buffer objects binding points
const uint PER_FRAME_UBO_BINDING            = 0;
const uint LIGHT_UBO_BINDING                = 1;
const uint MESH_MATERIAL_ARRAY_BINDING      = 2;
const uint POSITION_ARRAY_BINDING           = 3;

// Sampler binding points
const uint UNUSED_TEXTURE                   = 0;
const uint RENDER_TEXTURE                   = 1;

// Max values
const uint MAX_TEXTURE_ARRAYS               = 10;
const uint NUM_OBJECTS_MAX                  = 500;
const uint NUM_MESHES_MAX                   = 500;
const uint MAX_POINT_LIGHTS                 = 8;

#endif  // COMMON_H