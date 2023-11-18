typedef struct {
    float2 position;
    int radius;
    uint color;
} Particle;

bool pixel_in_image(int x,int y, int x_range, int y_range){
    return x >= 0 && x < x_range && y >= 0 && y < y_range;
}

bool pixel_in_circle(int x, int y, int2 center, int radius){
    int x_dist = x - center.x;
    int y_dist = y - center.y;
    return x_dist * x_dist + y_dist * y_dist < radius * radius;
}

uint4 get_particle_color_as_uint4(Particle particle){
    uint color = particle.color;
    uchar4 split_color = vload4(0, (uchar*)(&color));
    int4 new_color = convert_int4(split_color);
    return new_color;
}

__kernel void render_particles(
    __global Particle* particles, __write_only image2d_t output_image
){
    int2 image_shape = get_image_dim(output_image);
    int partical_index = get_global_id(0);
    Particle particle = particles[partical_index];
    float2 center_pixel_float = (float2)(((float)image_shape.x) * particle.position.x , ((float)image_shape.y) * particle.position.y);
    int2 center_pixel = (int2)(round(center_pixel_float.x), round(center_pixel_float.y));

    int2 x_pixel_range = (int2)(center_pixel.x - particle.radius, center_pixel.x + particle.radius);
    int2 y_pixel_range = (int2)(center_pixel.y - particle.radius, center_pixel.y + particle.radius);
    for(
        int x = x_pixel_range.x; x < x_pixel_range.y; x++
    ){
        for(int y = y_pixel_range.x; y < y_pixel_range.y; y++){
            if(pixel_in_image(x,y,image_shape.x, image_shape.y) && pixel_in_circle(x,y,center_pixel, particle.radius)){
                uint4 color = get_particle_color_as_uint4(particle);
                write_imageui(output_image, (int2)(x,y), color);
            }
        }
    }
}
