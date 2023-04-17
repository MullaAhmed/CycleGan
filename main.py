from options.test_options import TestOptions

from models import create_model
import torchvision.transforms as transforms
from PIL import Image
from util import util 
import os




opt = TestOptions().parse()  # get test options

# hard-code some parameters for test

opt.gpu_ids=[]
opt.model= "test"
opt.no_dropout = True   # no dropout for the generator
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.


def stylise(style,img):
    # style="style_ukiyoe"
    # img = Image.open("./input/Me.jpg")

    opt.name= style
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)   
    
    dataset=[
        {"A":img_tensor,
         'A_paths': ['Me.jpg']
        }
    ]

    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

   
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
       
        for label, im_data in visuals.items():
           
            im = util.tensor2im(im_data)
            image_pil = Image.fromarray(im)
            if label=="fake":
                # image_pil.show()
                return image_pil
            # util.save_image(im, save_path, aspect_ratio=1.0)
        

# style="style_ukiyoe"
# img = Image.open("./input/Me.jpg")
# stylise(style,img)