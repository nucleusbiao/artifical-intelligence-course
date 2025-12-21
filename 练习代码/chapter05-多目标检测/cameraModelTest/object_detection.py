import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional


class DetectionResult:
    def __init__(self):
        self.position = (0, 0)  # (u, v) - 检测框底部中点坐标
        self.confidence = 0.0
        self.bbox = (0, 0, 0, 0)  # (x, y, width, height) - 完整边界框信息
        self.class_id = 0  # 类别ID
        self.class_name = ""  # 类别名称


class SimpleFootballDetector:
    def __init__(self, model_path: str, confidence_threshold=0.25, nms_threshold=0.4):
        self.model_path = model_path
        self.confidence_ = confidence_threshold
        self.nms_threshold_ = nms_threshold

        # 您的自定义类别映射
        self.class_names = {
            0: "Ball",
            1: "Goalpost",
            2: "Person",
            3: "LCross",
            4: "TCross",
            5: "XCross",
            6: "PenaltyPoint",
            7: "Opponent",
            8: "BRMarker"
        }
        self.num_classes = len(self.class_names)

        print(f"正在加载ONNX模型: {model_path}")

        # 初始化ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            # 获取输出名称
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"模型加载成功! 输入名称: {self.input_name}, 输出名称: {self.output_names}")

            # 打印模型输入输出信息
            for i, input_info in enumerate(self.session.get_inputs()):
                print(f"输入 {i}: {input_info.name}, 形状: {input_info.shape}")
            for i, output_info in enumerate(self.session.get_outputs()):
                print(f"输出 {i}: {output_info.name}, 形状: {output_info.shape}")

        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")

    def inference(self, img: np.ndarray) -> List[DetectionResult]:
        # 预处理图像
        input_img = self.preprocess(img)

        # 运行推理
        print("正在进行推理...")
        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        # 处理输出
        if len(outputs) == 1:
            output_data = outputs[0]
        else:
            output_data = outputs[0]

        print(f"推理完成! 输出形状: {output_data.shape}")

        # 后处理 - 返回所有类别的检测结果
        detections = self.postprocess(output_data, img.shape)
        return detections

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # 调整大小并归一化
        input_size = (640, 640)  # YOLOv8默认输入尺寸
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0

        # 转换通道顺序: HWC to NCHW
        input_img = np.transpose(normalized, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        return input_img

    def nms(self, boxes: List[Tuple], scores: List[float], class_ids: List[int]) -> List[int]:
        """非极大值抑制算法"""
        if len(boxes) == 0:
            return []

        # 将边界框转换为(x1, y1, x2, y2)格式
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]  # 按置信度降序排序

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算当前框与剩余框的IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU低于阈值的框
            inds = np.where(iou <= self.nms_threshold_)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> List[DetectionResult]:
        """
        处理YOLOv8输出格式
        输出形状: (1, 13, 8400) - 13 = 4(bbox) + 9(classes)
        """
        orig_h, orig_w = orig_shape[:2]
        input_size = 640

        # YOLOv8输出格式: (1, 13, 8400)
        # 13 = 4(bbox) + 9(classes)
        output_data = outputs  # 形状: (1, 13, 8400)
        print(f"原始输出形状: {output_data.shape}")

        # 转置为更易处理的格式: (8400, 13)
        predictions = output_data.transpose(0, 2, 1).squeeze(0)
        print(f"转换后预测形状: {predictions.shape}")

        all_detections = []
        boxes = []
        scores = []
        class_ids = []

        # 遍历所有检测框 (8400个)
        for i in range(predictions.shape[0]):
            prediction = predictions[i]

            # 前4个值是边界框坐标 (x_center, y_center, width, height)
            bbox = prediction[0:4]

            # 剩余的值是9个类别的概率
            class_scores = prediction[4:4 + self.num_classes]

            # 找到最高类别分数和对应的类别ID
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # 应用置信度阈值
            if confidence > self.confidence_:
                x_center, y_center, w, h = bbox

                # 坐标转换到原始图像尺寸
                x_center_orig = (x_center / input_size) * orig_w
                y_center_orig = (y_center / input_size) * orig_h
                w_orig = (w / input_size) * orig_w
                h_orig = (h / input_size) * orig_h

                # 计算边界框的左上角坐标
                x1 = int(max(0, x_center_orig - w_orig / 2))
                y1 = int(max(0, y_center_orig - h_orig / 2))
                w_int = int(w_orig)
                h_int = int(h_orig)

                # 计算底部中点坐标 (u, v)
                u = int(x_center_orig)
                v = int(y1 + h_int)

                all_detections.append({
                    'position': (u, v),
                    'confidence': confidence,
                    'bbox': (x1, y1, w_int, h_int),
                    'class_id': class_id
                })

                boxes.append((x1, y1, w_int, h_int))
                scores.append(float(confidence))
                class_ids.append(class_id)

        # 如果没有检测到任何目标，返回空列表
        if len(all_detections) == 0:
            print("未检测到任何目标")
            return []

        print(f"应用NMS前检测到 {len(all_detections)} 个目标")

        # 应用NMS
        keep_indices = self.nms(boxes, scores, class_ids)

        # 构建最终结果
        results = []
        for idx in keep_indices:
            detection = all_detections[idx]

            result = DetectionResult()
            result.position = detection['position']
            result.confidence = detection['confidence']
            result.bbox = detection['bbox']
            result.class_id = detection['class_id']
            result.class_name = self.class_names.get(detection['class_id'], f"Class_{detection['class_id']}")

            results.append(result)

            print(f"检测结果: {result.class_name} 位置{result.position}, 置信度{result.confidence:.3f}")

        print(f"经过NMS后保留 {len(results)} 个检测框")
        return results

    def draw_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        result_image = image.copy()

        if not detections:
            print("没有检测结果可绘制")
            return result_image

        # 为不同类别定义不同颜色
        colors = {
            0: (0, 255, 0),  # Ball - 绿色
            1: (255, 0, 0),  # Goalpost - 蓝色
            2: (0, 0, 255),  # Person - 红色
            3: (255, 255, 0),  # LCross - 青色
            4: (255, 0, 255),  # TCross - 洋红色
            5: (0, 255, 255),  # XCross - 黄色
            6: (128, 0, 128),  # PenaltyPoint - 紫色
            7: (0, 128, 128),  # Opponent - 橄榄色
            8: (128, 128, 0)  # BRMarker - 深黄色
        }

        for detection in detections:
            # 获取类别对应的颜色
            color = colors.get(detection.class_id, (255, 255, 255))  # 默认白色

            # 绘制检测框
            x, y, w, h = detection.bbox
            cv2.rectangle(result_image,
                          (x, y),
                          (x + w, y + h),
                          color, 2)

            # 绘制底部中点
            u, v = detection.position
            cv2.circle(result_image, (u, v), 5, color, -1)

            # 构建显示文本
            display_text = f"{detection.class_name}: {detection.confidence:.2f}"

            # 绘制文本背景
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            # 文本位置
            text_bg_x1 = x
            text_bg_y1 = max(0, y - text_size[1] - 10)
            text_bg_x2 = x + text_size[0] + 5
            text_bg_y2 = max(0, y - 5)

            # 绘制文本背景
            cv2.rectangle(result_image,
                          (text_bg_x1, text_bg_y1),
                          (text_bg_x2, text_bg_y2),
                          color, -1)

            # 绘制文本
            text_org = (x + 2, max(15, y - 5))
            cv2.putText(result_image, display_text, text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            print(f"绘制检测结果: {detection.class_name} 位置({u},{v}), 置信度{detection.confidence:.3f}")

        return result_image


def test_football_detection(image_path: str, model_path: str, confidence_threshold=0.25, nms_threshold=0.4):
    print("=== YOLOv8多类别目标检测测试 ===")
    print(f"置信度阈值: {confidence_threshold}, NMS阈值: {nms_threshold}")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功! 尺寸: {image.shape[1]} × {image.shape[0]}")

    try:
        # 初始化检测器
        detector = SimpleFootballDetector(model_path,
                                          confidence_threshold=confidence_threshold,
                                          nms_threshold=nms_threshold)

        # 进行检测
        detections = detector.inference(image)

        if detections:
            print(f"成功检测到 {len(detections)} 个目标!")

            # 按类别统计
            class_count = {}
            for detection in detections:
                u, v = detection.position
                print(f"- {detection.class_name}: 位置({u}, {v}), 置信度{detection.confidence:.3f}")

                # 统计各类别数量
                if detection.class_name in class_count:
                    class_count[detection.class_name] += 1
                else:
                    class_count[detection.class_name] = 1

            print("\n检测结果统计:")
            for class_name, count in class_count.items():
                print(f"  {class_name}: {count}个")
        else:
            print("未检测到任何目标")

        # 绘制检测结果
        result_image = detector.draw_detections(image, detections)

        # 显示结果
        cv2.imshow("YOLOv8 Multi-Class Detection", result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = "yolov8_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请替换为实际的文件路径
    image_path = "football_image.jpg"
    model_path = "model/best.onnx"

    # 使用指定的置信度阈值和NMS阈值
    test_football_detection(image_path, model_path,
                            confidence_threshold=0.25,
                            nms_threshold=0.4)