#ifndef MESH_H
#define MESH_H

namespace Mesh {

  Mesh* loadObj(const std::string filename) {
    std::ifstream in(filename.c_str());

    if(!in.good())
    {
      std::cout  << "ERROR: loading obj:(" << filename << ") \n";
      exit(0);
    }

    char buffer[256], str[255];
    float f1,f2,f3;

    std::vector<glm::vec3> verts;
    std::vector<Triangle> faces;

    while(!in.getline(buffer,255).eof())
    {
      buffer[255] = '\0';

      sscanf_s(buffer,"%s",str,255);

      // reading a vertex
      if (buffer[0]=='v' && (buffer[1]==' '  || buffer[1]==32) )
      {
        if ( sscanf(buffer,"v %f %f %f",&f1,&f2,&f3)==3 )
        {
          verts.push_back(glm::vec3(f1,f2,f3));
        }
        else
        {
          std::cout << "ERROR: vertex format" << "\n";
          exit(-1);
        }
      }

      // reading FaceMtls 
      else if (buffer[0]=='f' && (buffer[1]==' ' || buffer[1]==32) )
      {
        Triangle f;
        int nt = sscanf(buffer,"f %d %d %d",&f.m_v.x,&f.m_v.y,&f.m_v.z);
        if( nt!=3 )
        {
          std::cout << "ERROR: FaceMtl format" << "\n";
          exit(-1);
        }
      
        faces.push_back(f);
      }
    }
  
    // construct Mesh

    Mesh* mesh = (Mesh*)malloc(sizeof(Mesh));

    size_t vertsMemSize = verts.size()*sizeof(glm::vec3);
    mesh->m_verts = (glm::vec3*)malloc(vertsMemSize);
    memcpy(mesh->m_verts, verts.data(), vertsMemSize);
    mesh->m_numVerts = verts.size();

    size_t facesMemSize = verts.size()*sizeof(Triangle);
    mesh->m_faces = (Triangle*)malloc(vertsMemSize);
    memcpy(mesh->m_faces, faces.data(), facesMemSize);
    mesh->m_numFaces = faces.size();

    std::printf("Loaded \"%s\" %u verts %u faces", filename.c_str(), verts.size(), faces.size());

    return mesh;
  }

}

#endif  // MESH_H