#ifndef FULLSCREENQUAD_H
#define FULLSCREENQUAD_H

#include "common.h"

class FullScreenQuad
{
private:
    GLuint fullScreenVertexArray;

public:

    void begin()
    {
        // Create buffer objects and vao for a full screen quad
        const uint numVertices = 4;
        glm::vec2 vertices[numVertices];
        vertices[0] = glm::vec2(-1.0, -1.0);
        vertices[1] = glm::vec2(1.0, -1.0);
        vertices[2] = glm::vec2(1.0, 0.75);
        vertices[3] = glm::vec2(-1.0, 1.0);

        const uint numElements = 6;
        unsigned short elements[numElements] = {0, 1, 2, 2, 3, 0};

        GLuint vertexBuffer;
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*numVertices, vertices, GL_STATIC_DRAW);

        GLuint elementArrayBuffer;
        glGenBuffers(1, &elementArrayBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementArrayBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short)*numElements, elements, GL_STATIC_DRAW);

        glGenVertexArrays(1, &fullScreenVertexArray);
        glBindVertexArray(fullScreenVertexArray);

        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glEnableVertexAttribArray(POSITION_ATTR);
        glVertexAttribPointer(POSITION_ATTR, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementArrayBuffer);

        //Unbind everything
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    void display()
    {
        glBindVertexArray(fullScreenVertexArray);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);
    }

    void displayInstanced(uint numInstances)
    {
        glBindVertexArray(fullScreenVertexArray);
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, numInstances);
        glBindVertexArray(0);
    }
};



#endif  // FULLSCREENQUAD_H