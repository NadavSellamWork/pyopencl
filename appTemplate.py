import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

class OpenGlRenderingEngine:
    def __init__(self, width, height):
        self.screenWidth = width
        self.screenHeight = height

        
        glEnable(GL_BLEND) # this enables the attribute of rendering multiple alpha components one on the other
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # this setting sets how open gl will blend colors based on alpha

        
        self.createQuad()
        self.createColorBuffer()


        #general OpenGL configuration
        self.shader = self.createShader()
        #define top and bottom points for a line segment
        glUseProgram(self.shader)
    
    def createQuad(self):
        # this quad is a plane that will contain the texture that will be rendered to the screen
        # x, y, z, s, t
        self.vertices = np.array(
            ( 1.0,  1.0, 0.0, 1.0, 1.0, #top-right
             -1.0,  1.0, 0.0, 0.0, 1.0, #top-left
             -1.0, -1.0, 0.0, 0.0, 0.0, #bottom-left
             -1.0, -1.0, 0.0, 0.0, 0.0, #bottom-left
              1.0, -1.0, 0.0, 1.0, 0.0, #bottom-right
              1.0,  1.0, 0.0, 1.0, 1.0), #top-right
             dtype=np.float32
        )

        self.vertex_count = 6

        """
        the vao is the object that contains the vbo.
        the vbo is raw data
        the vao contains the raw data using the vbo and contains metadata that describes how to interpret the data
        """
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # declare attribute of index 0 in the vao
        glEnableVertexAttribArray(0)
        # it describes the attribute of index 0 in the vao, it has 3 float numbers, not to be normalized, 20 is the stride (meaning how many bytes to skip forward to get to the next attribute in the next vertex) and ctypes.c_void_p(0) is the offset from the start of the object.
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    
    def createColorBuffer(self):
        # # the buffer data is a 1D array of 32 bit unsigned integers
        colorBufferData = np.array(
            [np.uint32((0 << 24) + (0 << 16) + (0 << 8) + (255 << 0)) for pixel in range(self.screenWidth * self.screenHeight)],
            dtype=np.uint32
        )

        # the color buffer is a buffer that can be used in the shaders, and have data from the array
        self.colorBuffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.colorBuffer)

        # configuring how to sample the texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # this command moves data from the np array in the cpu to the color buffer in the gpu
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.screenWidth,self.screenHeight,0,GL_RGBA,GL_UNSIGNED_INT_8_8_8_8,bytes(colorBufferData))
    
    def createShader(self):
        """
            Read source code, compile and link shaders.
            Returns the compiled and linked program.
        """

        vertex_src = """
#version 330 core
#extension GL_ARB_separate_shader_objects : enable

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTextureCoordinate;

layout (location=0) out vec2 fragmentTextureCoordinate;

void main()
{
    gl_Position = vec4(vertexPos, 1.0);
    fragmentTextureCoordinate = vertexTextureCoordinate;
}
"""

        fragment_src = """
#version 330 core
#extension GL_ARB_separate_shader_objects : enable

layout (location=0) in vec2 fragmentTextureCoordinate;

uniform sampler2D framebuffer;

out vec4 finalColor;

void main()
{
    finalColor = texture(framebuffer, fragmentTextureCoordinate);
}
"""
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader
    
    def set_color_buffer_data_from_array(self, data: np.ndarray):
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 1
        assert data.shape[0] == self.screenWidth * self.screenHeight
        assert data.dtype == np.uint32
        glBindTexture(GL_TEXTURE_2D, self.colorBuffer)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.screenWidth,self.screenHeight,0,GL_RGBA,GL_UNSIGNED_INT_8_8_8_8,bytes(data))
        
    def drawScreen(self):
        glBindTexture(GL_TEXTURE_2D, self.colorBuffer)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        pg.display.flip()
    
    def destroy(self):
        """
            Free any allocated memory
        """
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
        glDeleteTextures(1, (self.colorBuffer,))
        glDeleteProgram(self.shader)

class AppTemplate:
    """
        Calls high level control functions (handle input, draw scene etc)
    """

    def __init__(self):
        pg.init()
        self.screenWidth = 640
        self.screenHeight = 480
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((self.screenWidth, self.screenHeight), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        self.graphicsEngine = OpenGlRenderingEngine(self.screenWidth, self.screenHeight)
        self.graphicsEngine.drawScreen()
    
    def mainLoop(self):
        running = True
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        image_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
        data = np.zeros((self.screenWidth, self.screenHeight, 4), dtype=np.uint8)
        image = cl.Image(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, image_format, hostbuf=data)
        i = 0
        while (running):
            #events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            

            # r_data = (np.random.randn(self.screenWidth, self.screenHeight) * 255).astype(np.uint32)
            # g_data = (np.random.randn(self.screenWidth, self.screenHeight) * 255).astype(np.uint32)
            # b_data = (np.random.randn(self.screenWidth, self.screenHeight) * 255).astype(np.uint32)
            # a_data = (np.ones((self.screenWidth, self.screenHeight), dtype=np.uint32) * 255)
            r_data = (np.zeros((self.screenWidth, self.screenHeight))).astype(np.uint32) + i
            g_data = (np.zeros((self.screenWidth, self.screenHeight))).astype(np.uint32)
            b_data = (np.zeros((self.screenWidth, self.screenHeight))).astype(np.uint32)
            a_data = (np.ones((self.screenWidth, self.screenHeight), dtype=np.uint32) * 255)
            data = ((r_data << 24) + (g_data << 16) + (b_data << 8) + a_data).view(np.uint8).reshape(self.screenWidth, self.screenHeight, 4)

            copy_to_gpu_task = cl.enqueue_copy(queue, image, data, origin=(0, 0), region=(self.screenWidth, self.screenHeight))
            copy_to_cpu_task = cl.enqueue_copy(queue, data, image, origin=(0, 0), region=(self.screenWidth, self.screenHeight))
            cl.wait_for_events([copy_to_gpu_task, copy_to_cpu_task])
            self.graphicsEngine.set_color_buffer_data_from_array(data.view(np.uint32).reshape(-1))

            #render
            self.graphicsEngine.drawScreen()

            #timing
            self.clock.tick()
            framerate = int(self.clock.get_fps())
            pg.display.set_caption(f"Running at {framerate} fps.")
        self.quit()
    
    def quit(self):
        self.graphicsEngine.destroy()
        pg.quit()

if __name__ == "__main__":
    myApp = AppTemplate()
    myApp.mainLoop()
