__kernel void update_image(
    __read_only image2d_t input_image, __write_only image2d_t output_image
){
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    uint4 pixel = read_imageui(input_image, coord);
    uint4 next_pixel = (uint4)((pixel.x + 1 ) % 255, pixel.y, pixel.z, pixel.w);
    write_imageui(output_image, coord, next_pixel);
}
