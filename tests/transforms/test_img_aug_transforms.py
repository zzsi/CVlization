import numpy as np
from unittest.mock import patch
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from cvlization.transforms.img_aug_transforms import ImgAugTransforms, TRANSFORM_MENU

config_path = "tests/transforms/example_imgaug_config.json"
test_image_path = "tests/transforms/example_image.jpeg"


def return_segmap(image):
    segmap = np.zeros((128, 128, 1), dtype=np.int32)
    segmap[28:71, 35:85, 0] = 1
    segmap[10:25, 30:45, 0] = 2
    segmap[10:25, 70:85, 0] = 3
    segmap[10:110, 5:10, 0] = 4
    segmap[118:123, 10:110, 0] = 5
    return segmap


def bbox():
    bbox = [290, 115, 405, 385]
    bbox = torch.tensor(bbox, dtype=torch.int).unsqueeze(dim=0)
    return bbox


def test_imgaug_transform_creation_from_config_path():
    image_aug_object = ImgAugTransforms(cv_task=None, config_file=config_path)
    assert image_aug_object is not None
    assert image_aug_object.config["cv_task"] == "detection"
    assert len(image_aug_object.aug) == 5


def test_load_image():
    img = read_image(test_image_path)
    assert img is not None


def test_draw_bbox():
    img = read_image(test_image_path)
    img = draw_bounding_boxes(image=img, boxes=bbox(), width=3)
    assert img.shape == (3, 871, 840)
    img = torchvision.transforms.ToPILImage()(img)
    # Uncomment the following line to produce expected test case.
    # If a new image is generated, please visually inspect it.
    # img.save(test_image_path.replace(".jpeg", "_bbox.png"))
    expected_img = np.array(Image.open(test_image_path.replace(".jpeg", "_bbox.png")))
    img = np.array(img)
    assert expected_img.shape == img.shape
    assert expected_img.dtype == img.dtype
    np.testing.assert_array_equal(
        expected_img,
        img,
    )


def test_img_aug_classification():
    img = read_image(test_image_path)
    image_aug_object = ImgAugTransforms(
        cv_task="classification", config_file=config_path
    )
    image_aug_object.aug.seed_(1)
    with patch.object(image_aug_object.aug, "to_deterministic") as mock_fn:
        image_aug = image_aug_object(image=img)
        assert mock_fn.is_called
    # TODO: test the behavior when to_detereministic is not called.
    image_aug = image_aug_object(image=img)
    assert image_aug.shape == (3, 818, 645)
    image_aug = image_aug.transpose(1, 2, 0)
    image_aug = Image.fromarray(image_aug)
    assert image_aug.height == 818
    assert image_aug.width == 645
    # Uncomment the following line to produce expected test case.
    # If a new image is generated, please visually inspect it.
    # image_aug.save(test_image_path.replace(".jpeg", "_aug.png"))
    expected_img = np.array(Image.open(test_image_path.replace(".jpeg", "_aug.png")))
    np.testing.assert_array_equal(
        expected_img,
        image_aug,
    )


def test_img_aug_objdet():
    image_aug_object = ImgAugTransforms(cv_task=None, config_file=config_path)
    assert image_aug_object.config["cv_task"] == "detection"
    img = read_image(test_image_path)
    target = {}
    target["boxes"] = bbox()
    target["labels"] = torch.FloatTensor(np.ones(1))
    image_aug_object.aug.seed_(1)
    assert isinstance(img, torch.Tensor)
    image_aug, target = image_aug_object(image=img, target=target)
    image_tensor = torch.tensor(image_aug.astype(np.uint8))
    boxes_tensor = torch.tensor(target["boxes"], dtype=torch.int)
    image_aug = draw_bounding_boxes(
        image=image_tensor,
        boxes=boxes_tensor,
        width=3,
    )
    assert image_aug.shape == (3, 818, 645)
    image_aug = image_aug.numpy().transpose(1, 2, 0)
    # Uncomment the following line to produce expected test case.
    # If a new image is generated, please visually inspect it.
    # Image.fromarray(image_aug).save(test_image_path.replace(".jpeg", "_objdet_aug.png"))
    expected_img = np.array(
        Image.open(test_image_path.replace(".jpeg", "_objdet_aug.png"))
    )
    np.testing.assert_array_equal(
        expected_img,
        image_aug,
    )


def test_draw_semseg_map():
    img = read_image(test_image_path)
    img = np.moveaxis(img.numpy(), 0, -1)
    segmap = return_segmap(image=img)
    segmap = SegmentationMapsOnImage(segmap, shape=img.shape)
    img = segmap.draw_on_image(image=img)[0]
    assert img.shape == (871, 840, 3)
    # Uncomment the following line to produce expected test case.
    # If a new image is generated, please visually inspect it.
    # Image.fromarray(img).save(test_image_path.replace(".jpeg", "_segmap.png"))
    expected_img = np.array(Image.open(test_image_path.replace(".jpeg", "_segmap.png")))
    np.testing.assert_array_equal(
        expected_img,
        img,
    )


def test_img_aug_segmentation():
    image_aug_object = ImgAugTransforms(cv_task="semseg", config_file=config_path)
    img = read_image(test_image_path)
    segmap = return_segmap(image=img)
    image_aug_object.aug.seed_(1)
    image_aug, segmap = image_aug_object(image=img, segmap=segmap)
    image_aug = np.moveaxis(image_aug, 0, -1)
    image_aug = segmap.draw_on_image(image=image_aug)[0]
    # Uncomment the following line to produce expected test case.
    # If a new image is generated, please visually inspect it.
    # Image.fromarray(image_aug).save(test_image_path.replace(".jpeg", "_segmap_aug.png"))
    expected_img = np.array(
        Image.open(test_image_path.replace(".jpeg", "_segmap_aug.png"))
    )
    np.testing.assert_array_equal(
        expected_img,
        image_aug,
    )
