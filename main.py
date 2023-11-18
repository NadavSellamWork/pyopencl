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


        self.cl_current_image_buffer = cl.Image(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, image_format, hostbuf=data_1)
        self.cl_next_image_buffer = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, image_format, shape=(self.screenWidth, self.screenHeight))
        self.program = None
        self.load_shader()

        self.num_particles = 100
        particle_locations = np.random.rand(self.num_particles, 2).astype(np.float32)
        particle_radiuses = np.ones((self.num_particles), dtype=np.int32) * 5
        particles_data = np.empty(self.num_particles, dtype=[('position', np.float32, 2), ('radius', np.int32), ("color", np.int32)])
        particles_data["position"] = particle_locations
        particles_data["radius"] = particle_radiuses
        particles_data["color"] = np.random.randint(0, 2 ** 32 - 1, (self.num_particles), dtype=np.uint32)
        # notice ! that the compiler will not allow for a struct that that size is not a multiple of 8 (for access speeds)
        # so there will be padding, meaning that if the "color" attribute would not have been there, the compiler will pad the struct to be 16 bytes anyway
        # and will read the array wrong, because it will use a stride of 16.
        self.particles_data = particles_data

        self.particles_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=particles_data)
    
    def render_particles(self):
        kernel = self.program.render_particles
        kernel(self.queue, (self.num_particles, ), None, self.particles_buffer, self.cl_next_image_buffer)

    
    def load_shader(self):
        shader_code = read_file("cl_shaders/particles.cl")
        self.program = cl.Program(self.ctx, shader_code).build()
    
    def update(self):
        self.render_particles()
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
