"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path

import copy
import types
import os
import SimpleITK
import numpy as np
import scipy
import torch
import segmentation_models_pytorch_3d as smp
import gc
import math
from contextlib import ExitStack
from unittest.mock import patch
from glob import glob
from typing import Tuple, Dict, Any
from torch.utils.data import default_collate
from tqdm import tqdm

from convert_2d_to_3d import TimmUniversalEncoder3d
from convertors import create_rostepifanov_model, EvalFusing, voxel_sequential_selector


print('All required modules are loaded!!!')

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MIN_HU = -200
MAX_HU = 1200


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        # Hack with force_apply and targets to work with volumentations
        for t in self.transforms:
            data = t(force_apply=False, targets=['image'], **data)
            data.pop('force_apply', None)
            data.pop('targets', None)
        return data


class ConvertTypes:
    def __call__(self, **data):
        data['image'] = data['image'].astype(np.float32)
        return data


class NormalizeHu:
    def __init__(self, sub, div, clip=True):
        self.sub = sub
        self.div = div
        self.clip = clip
    
    def __call__(self, **data):
        data['image'] = (data['image'] - self.sub) / self.div
        if self.clip:
            data['image'] = np.clip(data['image'], 0.0, 1.0)
        return data


def crop_possiby_padded(arr, h_start, h_stop, w_start, w_stop, d_start, d_stop, patch_size):
    # Crop the array, but make sure that the indices are within the array
    arr = arr[
        h_start:min(h_stop, arr.shape[0]),
        w_start:min(w_stop, arr.shape[1]),
        d_start:min(d_stop, arr.shape[2]),
    ]

    # Pad if necessary
    if (
        arr.shape[0] < patch_size[0] or 
        arr.shape[1] < patch_size[1] or 
        arr.shape[2] < patch_size[2]
    ):
        arr = np.pad(
            arr,
            [
                (0, patch_size[0] - arr.shape[0]),
                (0, patch_size[1] - arr.shape[1]),
                (0, patch_size[2] - arr.shape[2]),
            ],
            mode='constant',
            constant_values=0,
        )

    return arr 


def generate_patches_3d(
    *arrays, 
    patch_size: Tuple[int, int, int] | None = None, 
    step_size: int = None,
):
    assert all(arr.ndim == 3 for arr in arrays)
    assert all(arr.shape == arrays[0].shape for arr in arrays)
    original_shape = arrays[0].shape

    if patch_size is None:
        yield arrays, (0, 0, 0), original_shape, original_shape
    else:
        if step_size is None:
            step_size = patch_size
        
        padded_shape = []
        for i in range(3):
            if (original_shape[i] - patch_size[i]) % step_size[i] == 0:
                padded_shape.append(original_shape[i])
            else:
                padded_shape.append(
                    ((original_shape[i] - patch_size[i]) // step_size[i] + 1) * step_size[i] + patch_size[i]
                )
        padded_shape = tuple(padded_shape)

        for h_start in range(0, padded_shape[0] - patch_size[0] + 1, step_size[0]):
            h_stop = h_start + patch_size[0]
            for w_start in range(0, padded_shape[1] - patch_size[1] + 1, step_size[1]):
                w_stop = w_start + patch_size[1]
                for d_start in range(0, padded_shape[2] - patch_size[2] + 1, step_size[2]):
                    d_stop = d_start + patch_size[2]

                    indices = (
                        (h_start, h_stop),
                        (w_start, w_stop),
                        (d_start, d_stop),
                    )
                    
                    image_patches = [
                        crop_possiby_padded(
                            arr, 
                            h_start, h_stop, 
                            w_start, w_stop, 
                            d_start, d_stop, 
                            patch_size
                        )
                        for arr in arrays
                    ]
                    
                    yield image_patches, indices, original_shape, padded_shape


def collate_fn(batch):
    output = dict()
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in ['name', 'indices', 'original_shape', 'padded_shape']:
            output[key] = values
        else:
            output[key] = default_collate(values)
            if key == 'image':
                output[key] = output[key][:, None, ...]
    return output


# https://github.com/bnsreenu/python_for_microscopists/blob/master/
# 229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
def spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2*(scipy.signal.windows.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.windows.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def spline_window_3d(h, w, d, power=2):
    h_wind = spline_window(h, power)
    w_wind = spline_window(w, power)
    d_wind = spline_window(d, power)
    return h_wind[:, None, None] * w_wind[None, :, None] * d_wind[None, None, :] 



class UnpatchifyMetrics:
    """Unpatchify predictions to original full image assuming the dataloader is sequential
    and calculate the metrics for each full image, then aggregate them.
    """
    def __init__(self, n_classes, metrics, save_dirpath):
        self.n_classes = n_classes
        self.metrics = metrics
        self.save_dirpath = save_dirpath
        if self.save_dirpath is not None:
            self.save_dirpath.mkdir(parents=True, exist_ok=True)
        self.name = None
        self.preds = None
        self.weigths = None
        self.original_shape = None 
        self.weight_kernel = None

    def _reset_current(self):
        self.name = None
        self.preds = None
        self.weights = None
        self.original_shape = None 
        self.weight_kernel = None

        # Cleanup heavy tensors
        gc.collect()

    def reset(self):
        self._reset_current()
        for metric in self.metrics.values():
            metric.reset()

    def _calculate_metrics(self):
        # Remove padding
        self.preds = self.preds[
            :,
            :self.original_shape[0], 
            :self.original_shape[1], 
            :self.original_shape[2]
        ]
        self.weights = self.weights[
            :self.original_shape[0], 
            :self.original_shape[1], 
            :self.original_shape[2]
        ]

        # Weighted average
        self.weights[self.weights == 0] = 1
        self.preds /= self.weights[None, ...]

    def _init(
        self, 
        name: str, 
        padded_shape: Tuple[int, int, int], 
        original_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
        device: torch.device = torch.device('cpu'),
    ):
        self.name = name
        self.preds = torch.zeros((self.n_classes, *padded_shape), dtype=torch.float32, device=device)
        self.weights = torch.zeros(padded_shape, dtype=torch.float32, device=device)
        self.original_shape = original_shape
        self.weight_kernel = torch.from_numpy(
            spline_window_3d(*patch_shape, power=2)
        ).to(device)

    def update(self, batch: Dict[str, Any]):
        """Update the metrics with the predictions of the batch.
        batch:
            - 'name': List[str]
            - 'padded_shape': List[Tuple[int, int, int]]
            - 'original_shape': List[Tuple[int, int, int]]
            - 'indices': List[Tensor[B, 3, 2]]
            - 'pred': FloatTensor[B, C, H, W, D], probabilities
        """
        B = batch['pred'].shape[0]
        for i in range(B):
            name: str = batch['name'][i]
            if self.name is None or name != self.name:
                if self.name is not None:
                    self._calculate_metrics()
                    self._reset_current()
                patch_shape = (
                    batch['indices'][i][0][1] - batch['indices'][i][0][0],
                    batch['indices'][i][1][1] - batch['indices'][i][1][0],
                    batch['indices'][i][2][1] - batch['indices'][i][2][0],
                )
                self._init(
                    name, 
                    batch['padded_shape'][i], 
                    batch['original_shape'][i],
                    patch_shape,
                    device=batch['pred'].device,
                )
            self.preds[
                :,
                batch['indices'][i][0][0]:batch['indices'][i][0][1],
                batch['indices'][i][1][0]:batch['indices'][i][1][1],
                batch['indices'][i][2][0]:batch['indices'][i][2][1],
            ] += (batch['pred'][i] * self.weight_kernel[None, ...])
            self.weights[
                batch['indices'][i][0][0]:batch['indices'][i][0][1],
                batch['indices'][i][1][0]:batch['indices'][i][1][1],
                batch['indices'][i][2][0]:batch['indices'][i][2][1],
            ] += self.weight_kernel

    def compute(self):
        # Last object
        if self.name is not None:
            self._calculate_metrics()
            self._reset_current()

        return {
            name: metric.aggregate().item()
            for name, metric in self.metrics.items()
        }


NOT_USED_BLOCK_NAMES = {'b_0_1', 'b_0_2', 'b_0_3'}
class Unetpp(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model = smp.UnetPlusPlus(**kwargs)
        
        del model.decoder.blocks
        
        blocks = {}
        skips = {}
        
        nblocks = 5
        
        in_channels = model.encoder.out_channels[1: ][::-1]
        skip_channels = (*in_channels[1:], 0)
        out_channels = in_channels
        
        for idx in range(nblocks):
            skips[f's_{idx+1}_{idx}'] = in_channels[-idx-1]
        
        for idx in range(nblocks):
            for jdx in range(nblocks - idx):
                depth = jdx
                layer = idx+jdx
                block_name = f'b_{depth}_{layer}'
                if block_name in NOT_USED_BLOCK_NAMES:
                    continue
        
                in_ = in_channels[-depth-1]
                skip_ = skip_channels[-depth-1]
                out_ = out_channels[-depth-1]
        
                if depth > 0:
                    for sdx in range(layer-depth):
                        skip_ += skips[f's_{depth}_{layer-sdx-2}']
        
                skips[f's_{depth}_{layer}'] = out_
        
                block = smp.decoders.unetplusplus.decoder.DecoderBlock(in_, skip_, out_)
                blocks[block_name] = block
        
            if idx == 0:
                in_channels = (0, *in_channels[:-1])
                skip_channels = (0, *skip_channels[:-2], 0)
        
        model.decoder.blocks = torch.nn.ModuleDict(blocks)
        model.decoder.nblocks = nblocks

        
        model.heads = torch.nn.ModuleDict()
        model.heads[f'{model.decoder.depth}'] = torch.nn.Conv3d(
            24,
            24,
            3,
            1,
            1
        )
        
        def decoder_forward(self, *feats):
            xs = dict()
        
            for idx, x in enumerate(feats):
                xs[f'x_{idx}_{idx-1}'] = x
        
            for idx in range(self.nblocks):
                for jdx in range(self.nblocks - idx):
                    depth = jdx
                    layer = idx+jdx
        
                    if depth == 0 and (layer != 0 and layer != self.nblocks-1):
                        continue
                    
                    block = self.blocks[f'b_{depth}_{layer}']
        
                    if depth == 0:
                        skip = None
                        shape = xs[f'x_{0}_{-1}'].shape
                    else:
                        skip = torch.concat([ xs[f'x_{depth}_{layer-sdx-1}'] for sdx in range(layer-depth+1) ], axis=1)
                        shape = xs[f'x_{depth}_{layer-1}'].shape
        
                    x = xs[f'x_{depth+1}_{layer}']
                    x = block(x, skip)
                    xs[f'x_{depth}_{layer}'] = x
                    if depth == 0 and layer == self.nblocks - 1:
                        return xs[f'x_{0}_{self.nblocks-1}']
        
            return xs[f'x_{0}_{self.nblocks-1}']
        
        def model_forward(self, x):
            f = self.encoder(x)
            x = self.decoder(*f)
            x = self.heads[f'{idx}'](x)
            return x
        
        # model.decoder.nblocks = len(model.decoder.blocks)
        
        model.decoder.forward = types.MethodType(decoder_forward, model.decoder)
        model.forward = types.MethodType(model_forward, model)

        del model.decoder.blocks['b_0_0'].attention1
        del model.decoder.blocks['b_0_4'].attention1

        self.model = model

    def forward(self, x):
        return self.model(x)


def encoder_name_to_patch_context_args(encoder_name):
    # SMP-3D encoders
    if not encoder_name.startswith('tu-'):
        return []

    # Custom 3D encoders
    if 'convnext' in encoder_name or 'efficientnet' in encoder_name:
        return [
            ('segmentation_models_pytorch_3d.encoders.TimmUniversalEncoder', TimmUniversalEncoder3d),
        ]
    else:
        raise ValueError(f'Unknown encoder_name {encoder_name}.')


def build_model(seg_arch, seg_kwargs):
    # Select segmentation model class
    seg_arch_to_class = {
        'smp.Unet': smp.Unet,
        'Unetpp': Unetpp,
    }
    seg_class = seg_arch_to_class[seg_arch]

    # Wrap segmentation model creation with context managers
    # to patch the model for 3D. CMs depend on the encoder_name
    encoder_name = seg_kwargs['encoder_name']
    context_args = encoder_name_to_patch_context_args(encoder_name)
    with ExitStack() as stack:
        _ = [stack.enter_context(patch(*args)) for args in context_args]
        model = seg_class(**seg_kwargs)

    return model


def run():
    # Read the input
    # Read the input
    image, spacing, direction, origin = load_image_file_as_array(
        location=INPUT_PATH / "images/ct-angiography",
    )
    
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    ############# Lines You can change ###########
    image = image.transpose(2, 1, 0)
    image = image[::-1, ::-1, :]

    ### crop
    oshape = image.shape
    oxsize, oysize, ozsize = oshape

    image = image[oxsize//4:-oxsize//4, oysize//4:-(oysize*2)//9]

    ### crop end

    transform = Compose(
        [
            ConvertTypes(),
            NormalizeHu(sub=MIN_HU, div=MAX_HU-MIN_HU, clip=True),
        ]
    )

    models = []

    # mkotyushev's models
    # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    ##### uncomment ############################################################

    saved_model_paths = [
        RESOURCE_PATH / "models" / "djgacrg6" / "epoch=9-step=4690.pt",  # fold 0 
        RESOURCE_PATH / "models" / "prumdud7" / "epoch=13-step=6566.pt",  # fold 1
        # RESOURCE_PATH / "models" / "2x7myvqv" / "epoch=9-step=4690.pt",  # fold 2
        # RESOURCE_PATH / "models" / "qamo2s3f" / "epoch=15-step=7504.pt",  # fold 3 
        # RESOURCE_PATH / "models" / "pioy56fm" / "epoch=12-step=6097.pt",  # fold 4 
    ]
    seg_arch = 'Unetpp'
    seg_kwargs = {
        'encoder_name': 'tu-tf_efficientnetv2_m.in21k_ft_in1k',
        'in_channels': 1,
        'classes': 24,
        'encoder_depth': 5,
        'encoder_weights': None,
    }
    should_not_contain = NOT_USED_BLOCK_NAMES | {f'heads.{i}' for i in range(4)}
    for saved_model_path in saved_model_paths:
        model = build_model(seg_arch, seg_kwargs)
        state_dict = torch.load(saved_model_path, map_location='cpu', weights_only=True)
        state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if all([not bk in k for bk in should_not_contain])}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.model.encoder.model = EvalFusing(model.model.encoder.model, 'timm-efficientnetv2-m')
        models.append(model)

    print("Defined the models...")

    # Inference
    metric = UnpatchifyMetrics(
        n_classes=24,
        metrics=dict(),
        save_dirpath=None,
    )
    # Full size by x, y
    # padded by 32 to comply with the SMP requirements
    patch_size = (
        math.ceil(image.shape[0] / 32) * 32,
        math.ceil(image.shape[1] / 32) * 32,
        256,
    )
    step_size = (64, 64, 128)  # x, y not used
    batch_size = 1
    bg_multiplier = 0.8
    batch = []

    def run_batch():
        nonlocal batch, metric, models, device, bg_multiplier
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            batch = collate_fn(batch)
            image = batch['image'].to(device)
            for model in models:
                pred = model(image)
                pred = torch.softmax(pred, dim=1)
                pred[:, 0] *= bg_multiplier
                pred = pred / pred.sum(dim=1, keepdim=True)
                pred = pred.cpu()
                metric.update({'pred': pred, **batch})
            batch = []

    for (
        (image_patch,), 
        indices, 
        original_shape, 
        padded_shape
    ) in tqdm(
        generate_patches_3d(
            image, patch_size=patch_size, step_size=step_size,
        )
    ):
        item = {
            'image': image_patch,
            'name': 'image',
            'indices': indices,
            'original_shape': original_shape,
            'padded_shape': padded_shape,
        }
        item = transform(**item)
        batch.append(item)

        if len(batch) == batch_size:
            run_batch()
    if len(batch) > 0:
        run_batch()
    metric._calculate_metrics()

    mkotyushev_prob_masks = metric.preds.cpu().float().numpy()
    del models
    torch.cuda.empty_cache()

    ########################################################################### 

    # aortic_branches = metric.preds.argmax(dim=0).to(torch.uint8).cpu().numpy()
    # aortic_branches = aortic_branches.transpose(2, 1, 0)

    # rostepifanov code
    bg_multiplier = 0.3

    from cc3d import connected_components

    def filter_largest_connected_component(masks, connectivity):
        filter_masks = np.array(masks)
        booled_masks = (masks > 0).astype(np.uint8)

        components, ncomponents = connected_components(booled_masks, connectivity=connectivity, return_N=True)

        if ncomponents > 0:
            sizes = [ np.sum(components == idx + 1) for idx in np.arange(ncomponents) ]
            filter_masks[components != (np.argmax(sizes) + 1)] = 0

        return filter_masks

    rostepifanov_models = []

    rostepifanov_saved_model_paths = [
        RESOURCE_PATH / "models" / "model_tranm_seed0.pth",
        RESOURCE_PATH / "models" / "model_tranm_seed1.pth",
        RESOURCE_PATH / "models" / "model_tranm_seed2.pth",
        RESOURCE_PATH / "models" / "model_tranm_seed3.pth",
    ]

    for saved_model_path in rostepifanov_saved_model_paths:
        model = create_rostepifanov_model()
        state_dict = torch.load(saved_model_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if all([not bk in k for bk in should_not_contain])}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        model.encoder = EvalFusing(model.encoder, 'timm-efficientnetv2-m')
        rostepifanov_models.append(model)

    imgs = image

    shape = imgs.shape
    xsize, ysize, _ = shape

    vxsize, vysize, vzsize = (160, 160, 70)
    sxsize, sysize, szsize = (160, 160, 35)

    voxel_shape = (xsize, ysize, vzsize)
    steps = (xsize, ysize, szsize)

    prob_masks = np.zeros((24, *shape), dtype=np.float32)
    count_masks = np.zeros(shape, dtype=np.uint8)

    for _, selector in tqdm([*voxel_sequential_selector(voxel_shape, ['x'], [imgs.shape], steps)]):
        with torch.no_grad():
            voxels_batch = imgs[selector]
            voxels_batch = (voxels_batch + 200) / 1000
            voxels_batch = np.clip(voxels_batch, a_min=0, a_max=1)
            voxels_batch = torch.tensor([voxels_batch[None]]).float().to(device)

            voxels_batch = voxels_batch

            for model in rostepifanov_models:
                logits_batch = model(voxels_batch)

                prob_masks_batch = logits_batch.softmax(dim=1)
                prob_masks_batch[:, 0] *= bg_multiplier
                prob_masks_batch = prob_masks_batch / prob_masks_batch.sum(dim=1, keepdim=True)
                prob_masks_batch = prob_masks_batch.cpu().numpy()

                prob_masks[(slice(0, None), *selector)] += prob_masks_batch[0]
                count_masks[selector] += 1

    prob_masks /= count_masks

    rostepifanov_prob_masks = prob_masks

    # ensemble

    prob_masks = 0.7 * rostepifanov_prob_masks + 0.3 * mkotyushev_prob_masks
    #prob_masks = rostepifanov_prob_masks
    # prob_masks = mkotyushev_prob_masks
    # prob_masks = (rostepifanov_prob_masks + mkotyushev_prob_masks) / 2
    # prob_masks = rostepifanov_prob_masks

    aortic_branches = prob_masks.argmax(axis=0).astype(np.uint8)

    # Post-processing
    for idx in tqdm(range(1, 24)):
        aortic_branches[aortic_branches == idx] = idx * filter_largest_connected_component(aortic_branches == idx, 6).astype(np.uint8)[aortic_branches == idx]

    ### PADDING ###

    padded_aortic_branches = np.zeros(oshape, dtype=np.uint8)
    padded_aortic_branches[oxsize//4:-oxsize//4, oysize//4:-(oysize*2)//9] = aortic_branches

    aortic_branches = padded_aortic_branches

    ### ###

    aortic_branches = aortic_branches[::-1, ::-1, :]
    aortic_branches = aortic_branches.transpose(2, 1, 0)

    ########## Don't Change Anything below this 
    # For some reason if you want to change the lines, make sure the output segmentation has the same properties (spacing, dimension, origin, etc) as the 
    # input volume

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/aortic-branches",
        array=aortic_branches,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
    )
    print('Saved!!!')
    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin




def write_array_as_image_file(*, location, array, spacing, origin, direction):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
