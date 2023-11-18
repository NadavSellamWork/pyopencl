from appTemplate import AppTemplate
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
class App(AppTemplate):
    def __init__(self):
        super().__init__()
        image_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        data_1 = np.zeros((self.screenWidth, self.screenHeight, 4), dtype=np.uint8)
        data_1[...,0] = 1
        data_1[...,-1] = 255
        data_1 = data_1.view(np.int32).squeeze()


        data_2 = (np.zeros((self.screenWidth, self.screenHeight, 4), dtype=np.uint8) + 2).view(np.int32).squeeze()
        self.cl_current_image_buffer = cl.Image(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, image_format, hostbuf=data_1)
        self.cl_next_image_buffer = cl.Image(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, image_format, hostbuf=data_2)
        self.program = None
        self.load_shader()
    
    def load_shader(self):
        shader_code = read_file("cl_shaders/simple_image_manipulation.cl")
        self.program = cl.Program(self.ctx, shader_code).build()
    
    def update(self):
        self.program.update_image(self.queue, (self.screenWidth, self.screenHeight), None, self.cl_current_image_buffer, self.cl_next_image_buffer)
        cl.enqueue_copy(self.queue, self.cl_current_image_buffer,self.cl_next_image_buffer, src_origin=(0, 0),dest_origin=(0,0), region=(self.screenWidth, self.screenHeight)).wait()
   
    def mainLoop(self):
        running = True
        data = np.zeros((self.screenWidth, self.screenHeight, 4), dtype=np.uint8)       
        while (running):
            #events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            
            self.update()
            cl.enqueue_copy(self.queue, data, self.cl_current_image_buffer, origin=(0, 0), region=(self.screenWidth, self.screenHeight)).wait()
            out = data.view(np.uint32).reshape(-1).byteswap(False) # the byteswap call changes to little indian 
            self.graphicsEngine.set_color_buffer_data_from_array(out)

            #render
            self.graphicsEngine.drawScreen()

            #timing
            self.clock.tick()
            framerate = int(self.clock.get_fps())
            pg.display.set_caption(f"Running at {framerate} fps.")
        self.quit()

if __name__ == "__main__":
    App().mainLoop()
