import sys
import os

# 将当前脚本所在项目的根目录（即 BuildingAgent）加入 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/.../model_api
project_root = os.path.dirname(current_dir)               # /home/.../BuildingAgent
sys.path.insert(0, project_root)

# 现在可以正常导入 TRELLIS 包
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils

class ImageTo3DConverter:
    def __init__(self, model_path):
        os.environ['SPCONV_ALGO'] = 'native'
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
        self.pipeline.cuda()

    def convert(self, image_path, seed=1):
        image = Image.open(image_path)
        outputs = self.pipeline.run(
            image,
            seed=seed,
        )
        return outputs

    @staticmethod
    def render_and_save(outputs, output_format='gaussian', filename='output.mp4'):
        video = render_utils.render_video(outputs[output_format][0])['color']
        imageio.mimsave(filename, video, fps=30, codec='libx264')

    @staticmethod
    def save_as_glb(gaussian_output, mesh_output, filename='output.glb', simplify=0.95, texture_size=1024):
        glb = postprocessing_utils.to_glb(
            gaussian_output,
            mesh_output,
            simplify=simplify,
            texture_size=texture_size,
        )
        glb.export(filename)

    @staticmethod
    def save_as_ply(gaussian_output, filename='output.ply'):
        gaussian_output.save_ply(filename)