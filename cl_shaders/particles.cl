__kernel void update_image(
    __read_only image2d_t input_image, __write_only image2d_t output_image
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    uint4 pixel = read_imageui(input_image, coord);
    uint4 next_pixel = (uint4)((pixel.x + 1 ) % 255, pixel.y, pixel.z, pixel.w);
    write_imageui(output_image, coord, next_pixel);
}


bool pixel_in_image(int x,int y, int x_range, int y_range){
    return x >= 0 && x < x_range && y >= 0 && y < y_range;
}

bool pixel_in_circle(int x, int y, int2 center, int radius){
    int x_dist = x - center.x;
    int y_dist = y - center.y;
    return x_dist * x_dist + y_dist * y_dist < radius * radius;
}

typedef struct {
    float2 position;
    int radius;
    int color;
} Particle;

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
                write_imageui(output_image, (int2)(x,y), (uint4)(255, 255,255,255));
            }
        }
    }
}
