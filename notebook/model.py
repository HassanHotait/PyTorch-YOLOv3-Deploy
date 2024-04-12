from pytorchyolo.models import Darknet
from pytorchyolo.utils.utils import weights_init_normal, non_max_suppression
from pytorchyolo.detect import _create_data_loader, _draw_and_save_output_image, load_classes
import torch
import tqdm
from torch.autograd import Variable

model_path = "C:\\Users\\Hasan\\OneDrive\\Desktop\\Projects\\PyTorch-YOLOv3-Deploy\\config\\yolov3.cfg"
device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
model = Darknet(model_path).to(device)

model.apply(weights_init_normal)   
model.load_darknet_weights("C:\\Users\\Hasan\\OneDrive\\Desktop\\Projects\\PyTorch-YOLOv3-Deploy\\weights\\yolov3.weights")

scripted_model = torch.jit.script(model)
scripted_model.save('model_store/yolo_deploy.pt')