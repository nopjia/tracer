Dillinger
Dropbox 
GitHub 
Utilities 
Select Your Theme 

32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
- **Pan** - Middle Click drag
- **Focal Length** - F / Shift+F
- **Lens Radius** - G / Shift+G
Object Manipulation
- **Translate** - Ctrl + Left Click +  drag
- **Scale** - Ctrl + Right Click +  drag
- **Rotate** - Ctrl + Middle Click +  drag
## Details
Real-time iterative path tracer in CUDA.
Image draw done through CUDA OpenGL Interop, by mapping OpenGL pixel buffer object 
to CUDA memory, then drawing the pixel buffer out as a texture on a full screen 
quad.
Path tracing kernel is a per-bounce structure, instead of a mega-kernel structure, 
in order to minimize thread divergence.
In addition, there is a fast visualization mode, done using simple single bounce 
ray tracing and arbitrary Blinn-Phong lighting.
Runs at ~20fps on GeForce GT 650M. Achieves considerably converged image at ~900 
iterations, after ~45 seconds.
## References
- [Matt Pharr, Physically Based Rendering](http://www.pbrt.org/)
- [CUDA Ray Tracing Tutorial](http://cg.alexandra.dk/2009/08/10/triers-cuda-ray
-tracing-tutorial/)
- [smallpt: Global Illumination in 99 lines of C++](http://www.kevinbeason.com
/smallpt/)
CUDA Path Tracer
CUDA Path Tracer
Advanced Rendering Seminar, University of Pennsylvania, Spring 2013

Features
Diffuse Shading
Diffuse Reflection
Specular Reflection
Fresnel Reflection and Refraction
Anti-aliasing
Interactive Depth of Field
Interactive Camera
Interactive Object Manipulation
Two visualization modes:

Fast single bounce ray tracing (default at startup)
Path tracing
Controls
Modes

Ray Tracer - Press 1
Path Tracer - Press 2
Camera Controls

Rotate - Left Click drag
Zoom - Right Click drag
Pan - Middle Click drag
Focal Length - F / Shift+F
Lens Radius - G / Shift+G
Object Manipulation

Translate - Ctrl + Left Click + drag
Scale - Ctrl + Right Click + drag
Rotate - Ctrl + Middle Click + drag
Details
Real-time iterative path tracer in CUDA.

Image draw done through CUDA OpenGL Interop, by mapping OpenGL pixel buffer object to CUDA memory, then drawing the pixel buffer out as a texture on a full screen quad.

Path tracing kernel is a per-bounce structure, instead of a mega-kernel structure, in order to minimize thread divergence.

In addition, there is a fast visualization mode, done using simple single bounce ray tracing and arbitrary Blinn-Phong lighting.

Runs at ~20fps on GeForce GT 650M. Achieves considerably converged image at ~900 iterations, after ~45 seconds.

References
Matt Pharr, Physically Based Rendering
CUDA Ray Tracing Tutorial
smallpt: Global Illumination in 99 lines of C++
A Raytracer in C++
Reflections and Refractions in Ray Tracing