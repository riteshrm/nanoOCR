import zipfile
from shapely.geometry import Polygon
import pyclipper
import os
from pathlib import Path
import shutil
import torch
import numpy as np
import math
from pathlib import Path
import shutil
import argparse
import cv2
import random
import json
from tqdm import tqdm
random.seed(42)

class RandomCropData():
    max_tries = 50
    min_crop_side_ratio =0.1
    require_original_image = False

    def __init__(self, width, height):
        self.size = (width, height)

    def process(self, data):
        img = data['image']
        ori_img = img
        ori_lines = data['polygons']

        all_care_polys = []

        for i in range(len(data['polygons'])):
            if not  data['ignore_tags'][i]:
                all_care_polys.append(data['polygons'][i])
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros(
            (self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(
            img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        lines = []
        kept_ignore_tags = []

        for polygon, ignore in zip(data["polygons"], data["ignore_tags"]):
            poly = (
                (np.asarray(polygon) - (crop_x, crop_y)) * scale
            ).tolist()

            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                lines.append(poly)
                kept_ignore_tags.append(ignore)

        data["polygons"] = np.asarray(lines, dtype=np.float32)
        data["ignore_tags"] = np.asarray(kept_ignore_tags, dtype=np.uint8)

        if self.require_original_image:
            data['image'] = ori_img
        else:
            data['image'] = img
        data['lines'] = ori_lines
        data['scale_w'] = scale
        data['scale_h'] = scale

        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        if len(axis):
            regions.append(axis[min_axis:])
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

class MakeSegDetectionData():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size =8
    shrink_ratio=0.4

    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # height = min(np.linalg.norm(polygon[0] - polygon[3]),
            #              np.linalg.norm(polygon[1] - polygon[2]))
            # width = min(np.linalg.norm(polygon[0] - polygon[1]),
            #             np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.


class MakeBorderMap():
    r'''
    Making the border map from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7

    def process(self, data, *args, **kwargs):
        r'''
        required keys:
            image, polygons, ignore_tags
        adding keys:
            thresh_map, thresh_mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2

class NormalizeImage():
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def process(self, data):
        assert 'image' in data, '`image` in data is required by this process'
        image = data['image']
        image -= self.RGB_MEAN
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data['image'] = image
        return data

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image

def resize_image(img, short_side=512):
    height, width, _ = img.shape
    if height < width:
        new_height = short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img
def tdtr(args):
    print("Processing TDTR")
    if args.tdtr_zip_file is not None:
        with zipfile.ZipFile(args.tdtr_zip_file) as z:
            z.extractall("TD_TR")
        src_root = Path("TD_TR/TD_TR")
        dst_root = Path(f'{args.save_path}/TDTR')

        dst_root.mkdir(parents=True, exist_ok=True)

        for name in ["TD500", "TR400"]:
            src = src_root / name

            if not src.exists():
                continue

            for item in src.iterdir():
                dst = dst_root / item.name

                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)

                elif item.suffix == ".txt":
                    if dst.exists():
                        existing = dst.read_text().rstrip("\n")
                        incoming = item.read_text().strip("\n")

                        dst.write_text(existing + "\n" + incoming + "\n")
                    else:
                        shutil.copy2(item, dst)

                else:
                    shutil.copy2(item, dst)

        shutil.rmtree("TD_TR")

        print("Merged into det_data and cleaned successfully!")

    images_dir = os.path.join(f'{args.save_path}/TDTR', 'train_images')
    gt_dir = os.path.join(f'{args.save_path}/TDTR', "train_gts")

    image_names = os.listdir(images_dir)
    num_val = max(1, round(len(image_names) * args.split))
    num_val = min(num_val, len(image_names) - 1)
    val_names = set(image_names[:num_val])

    np.random.seed(42)
    resizer = RandomCropData(args.height, args.width)
    segmenter = MakeSegDetectionData()
    border_map = MakeBorderMap()
    normalizer = NormalizeImage()

    train_dir = os.path.join(args.save_path, "processed_train")
    val_dir = os.path.join(args.save_path, "processed_val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for filename in tqdm(image_names):
        image_path = os.path.join(images_dir, filename)
        gt_path = os.path.join(gt_dir, filename + ".txt")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_shape = tuple(image.shape[:2])
        image_polygons = []
        image_ignore_tags = []

        with open(gt_path, encoding="utf-8-sig") as reader:
            for line in reader:
                if not line.strip():
                    continue

                parts = [part.strip() for part in line.strip().split(",")]
                label = parts[-1]

                polygon = np.asarray(
                    parts[:-1], dtype=np.float32
                ).reshape(-1, 2)

                image_polygons.append(polygon)

                image_ignore_tags.append(label in ("1", "###"))

        is_val = filename in val_names

        data = {
            "image": image.astype(np.float32),
            "polygons": image_polygons,
            "ignore_tags": np.asarray(image_ignore_tags, dtype=np.uint8),
            "filename": filename,
            "original_shape": original_shape,
            "is_training": not is_val,
        }
      

        if is_val:
            data["shape"] = original_shape
            data["image"] = resize_image(data["image"], short_side=512)
            output_dir = val_dir
        else:
            data = resizer.process(data)
            data["shape"] = tuple(data["image"].shape[:2])
            data = segmenter.process(data)
            data = border_map.process(data)
            output_dir = train_dir

        data = normalizer.process(data)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            elif key == "polygons":
                data[key] = [
                    torch.as_tensor(poly, dtype=torch.float32)
                    for poly in value
                ]

        output_path = os.path.join(output_dir, Path(filename).stem + ".pt")
        torch.save(data, output_path)
    print("completed")

def convert_textocr(args, data_folder, split_type):
    print(f"Processing TextOCR {split_type}")
    json_path = os.path.join(data_folder, f"{split_type}.json")
    images_folder = os.path.join(data_folder, "images")
    save_images_folder = os.path.join(data_folder, f"{split_type}_images")
    gts_folder = os.path.join(data_folder, f"{split_type}_gts")

    os.makedirs(save_images_folder, exist_ok=True)
    os.makedirs(gts_folder, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for image_id, image_info in data["imgs"].items():
        image_name = image_info["file_name"].split("/")[-1]
        source_image = os.path.join(images_folder, image_name)

        if not os.path.isfile(source_image):
            continue

        lines = []
        flag = False

        for annotation_id in data["imgToAnns"].get(image_id, []):
            coords = data["anns"][annotation_id]["points"]
            line = ",".join(str(round(value)) for value in coords) + ",0"
            if len(line.split(','))>9:
                flag=True
                continue
            lines.append(line)

        if flag:
            continue

        gt_name = image_name + ".txt"
        gt_path = os.path.join(gts_folder, gt_name)
        shutil.copy2(
                    source_image,
                    os.path.join(save_images_folder, image_name),
                )

        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    images_dir = save_images_folder
    gt_dir = gts_folder

    image_names = os.listdir(images_dir)

    np.random.seed(42)
    resizer = RandomCropData(args.height, args.width)
    segmenter = MakeSegDetectionData()
    border_map = MakeBorderMap()
    normalizer = NormalizeImage()

    pt_save_dir = os.path.join(args.save_path, f"processed_{split_type}")

    os.makedirs(pt_save_dir, exist_ok=True)

    for filename in tqdm(image_names):
        image_path = os.path.join(images_dir, filename)
        gt_path = os.path.join(gt_dir, filename + ".txt")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_shape = tuple(image.shape[:2])
        image_polygons = []
        image_ignore_tags = []

        with open(gt_path, encoding="utf-8-sig") as reader:
            for line in reader:
                if not line.strip():
                    continue

                parts = [part.strip() for part in line.strip().split(",")]
                label = parts[-1]

                polygon = np.asarray(
                    parts[:-1], dtype=np.float32
                ).reshape(-1, 2)

                image_polygons.append(polygon)

                image_ignore_tags.append(label in ("1", "###"))

        is_val = split_type=="val"

        data = {
            "image": image.astype(np.float32),
            "polygons": image_polygons,
            "ignore_tags": np.asarray(image_ignore_tags, dtype=np.uint8),
            "filename": filename,
            "original_shape": original_shape,
            "is_training": not is_val,
        }
        

        if is_val:
            data["shape"] = original_shape
            data["image"] = resize_image(data["image"], short_side=512)
        else:
            data = resizer.process(data)
            data["shape"] = tuple(data["image"].shape[:2])
            data = segmenter.process(data)
            data = border_map.process(data)

        data = normalizer.process(data)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            elif key == "polygons":
                data[key] = [
                    torch.as_tensor(poly, dtype=torch.float32)
                    for poly in value
                ]

        output_path = os.path.join(pt_save_dir, Path(filename).stem + ".pt")
        torch.save(data, output_path)
    print("completed")





def textocr(args):
    textocr_folder = os.path.join(args.save_path, "TextOCR")
    images_folder = os.path.join(textocr_folder, "images")

    os.makedirs(textocr_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    shutil.copy2('train.json',os.path.join(textocr_folder, 'train.json'))
    shutil.copy2('val.json',os.path.join(textocr_folder, 'val.json'))

    with zipfile.ZipFile(args.textocr_zip_file) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            filename = member.filename.replace("\\", "/").split("/")[-1]

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            destination = os.path.join(images_folder, filename)

            with archive.open(member) as source:
                with open(destination, "wb") as target:
                    shutil.copyfileobj(source, target)

    convert_textocr(args, f"{args.save_path}/TextOCR", "val")
    convert_textocr(args, f"{args.save_path}/TextOCR", "train")


def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--tdtr_zip_file", type=str, default='TD_TR.zip')
    parser.add_argument("--textocr_zip_file", type=str, default="train_val_images.zip")
    parser.add_argument("--save_path", type=str, default='det_data')
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--data_type", type=str, default="textocr")

    args = parser.parse_args()

    if args.data_type=='tdtr':
        tdtr(args)
    elif args.data_type=='textocr':
        textocr(args)
    elif args.data_type=='both':
        tdtr(args)
        textocr(args)
    else:
        print("Unknown data type")
    
if __name__=="__main__":
    main()