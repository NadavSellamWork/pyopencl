typedef struct {
    float2 position;
    int radius;
    uint color;
} Particle;

bool pixel_in_image(int2 location, int x_range, int y_range){
    return location.x >= 0 &&location.x < x_range && location.y >= 0 && location.y < y_range;
}

bool pixel_in_circle(int2 location, int2 center, int radius){
    float2 distance_vector = convert_float2(location - center);
    float suqared_distance = length(distance_vector) * length(distance_vector);
    return suqared_distance < radius * radius;
}

bool is_pixle_in_circle_boundary(int2 location, int2 center, int radius, float boundary_width){
    // this function not in use ...
    float2 distance_vector = convert_float2(location - center);
    float distance = length(distance_vector);
    float suqared_distance = distance * distance;
    float suqared_radius = radius * radius;
    float squared_boundary_width = boundary_width * boundary_width;
    return suqared_distance > suqared_radius && fabs(suqared_distance - suqared_radius) < squared_boundary_width;
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

    int2 x_pixel_range = (int2)(center_pixel.x - particle.radius - 3, center_pixel.x + particle.radius + 3);
    int2 y_pixel_range = (int2)(center_pixel.y - particle.radius - 3, center_pixel.y + particle.radius + 3);
    for(
        int x = x_pixel_range.x; x < x_pixel_range.y; x++
    ){
        for(int y = y_pixel_range.x; y < y_pixel_range.y; y++){
            int2 pixel_location = (int2)(x,y);
            if(pixel_in_image(pixel_location,image_shape.x, image_shape.y)){

                if(pixel_in_circle(pixel_location,center_pixel, particle.radius)){
                    uint4 color = get_particle_color_as_uint4(particle);
                    write_imageui(output_image, (int2)(x,y), color);
                }
            }
        }
    }
}
