import imageio
from image_to_3d_api import ImageTo3DConverter

def main():
    # 初始化模型路径和输入图像路径
    model_path = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/TRELLIS-image-large"
    image_path = "/home/jiangbaoyang/model_api/result.png"

    # 创建ImageTo3DConverter实例并进行转换
    converter = ImageTo3DConverter(model_path=model_path)
    outputs = converter.convert(image_path=image_path, seed=1)

    # 渲染并保存视频
    converter.render_and_save(outputs, output_format='gaussian', filename="sample_gs.mp4")
    converter.render_and_save(outputs, output_format='radiance_field', filename="sample_rf.mp4")
    converter.render_and_save(outputs, output_format='mesh', filename="sample_mesh.mp4")

    # 保存GLB和PLY文件
    converter.save_as_glb(outputs['gaussian'][0], outputs['mesh'][0], filename="sample.glb")
    converter.save_as_ply(outputs['gaussian'][0], filename="sample.ply")

if __name__ == "__main__":
    main()