import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import math


class DetectionResult:
    def __init__(self):
        self.position = (0, 0)  # (u, v) - 检测框底部中点坐标
        self.confidence = 0.0
        self.bbox = (0, 0, 0, 0)  # (x, y, width, height) - 完整边界框信息
        self.world_position = (0.0, 0.0, 0.0)  # (x, y, z) - 世界坐标系中的位置


class Point3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Point2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class CameraIntrinsics:
    def __init__(self, fx=500.0, fy=500.0, cx=640.0, cy=360.0,
                 k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2


class TransformMatrix:
    def __init__(self):
        self.data = np.eye(4, dtype=np.float64)

    def __matmul__(self, other):
        result = TransformMatrix()
        result.data = self.data @ other.data
        return result


class BackProjector:
    def __init__(self, intrinsics: CameraIntrinsics, extrinsic: TransformMatrix):
        self.intrinsics_ = intrinsics
        self.extrinsic_ = extrinsic
        self.calculate_inverse_extrinsic()

    def calculate_inverse_extrinsic(self):
        R = self.extrinsic_.data[0:3, 0:3]
        t = self.extrinsic_.data[0:3, 3]

        Rt_t = R.T @ t

        self.extrinsic_inv_ = TransformMatrix()
        self.extrinsic_inv_.data[0:3, 0:3] = R.T
        self.extrinsic_inv_.data[0:3, 3] = -Rt_t
        self.extrinsic_inv_.data[3, 3] = 1.0

    def pixel_to_image(self, pixel_point: Point2D) -> Point2D:
        x = pixel_point.x - self.intrinsics_.cx
        y = pixel_point.y - self.intrinsics_.cy
        return Point2D(x, y)

    def remove_distortion(self, distorted_point: Point2D) -> Point2D:
        x_distorted = distorted_point.x / self.intrinsics_.fx
        y_distorted = distorted_point.y / self.intrinsics_.fy

        x = x_distorted
        y = y_distorted

        for _ in range(10):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            radial_factor = 1.0 + self.intrinsics_.k1 * r2 + self.intrinsics_.k2 * r4 + self.intrinsics_.k3 * r6
            delta_x = 2 * self.intrinsics_.p1 * x * y + self.intrinsics_.p2 * (r2 + 2 * x * x)
            delta_y = self.intrinsics_.p1 * (r2 + 2 * y * y) + 2 * self.intrinsics_.p2 * x * y

            x_ideal = (x_distorted - delta_x) / radial_factor
            y_ideal = (y_distorted - delta_y) / radial_factor

            x = x_ideal
            y = y_ideal

        x_ideal = x * self.intrinsics_.fx
        y_ideal = y * self.intrinsics_.fy

        return Point2D(x_ideal, y_ideal)

    def image_to_camera_ray(self, ideal_point: Point2D) -> Point3D:
        x = ideal_point.x / self.intrinsics_.fx
        y = ideal_point.y / self.intrinsics_.fy
        return Point3D(x, y, 1.0)

    def camera_to_world(self, camera_point: Point3D) -> Point3D:
        camera_homogeneous = np.array([camera_point.x, camera_point.y, camera_point.z, 1.0])
        world_homogeneous = self.extrinsic_inv_.data @ camera_homogeneous

        if abs(world_homogeneous[3]) > 1e-10:
            return Point3D(
                world_homogeneous[0] / world_homogeneous[3],
                world_homogeneous[1] / world_homogeneous[3],
                world_homogeneous[2] / world_homogeneous[3]
            )
        else:
            return Point3D(
                world_homogeneous[0],
                world_homogeneous[1],
                world_homogeneous[2]
            )

    def estimate_ball_position(self, pixel_point: Point2D, verbose=False) -> Point3D:
        if verbose:
            print("=== 利用地面约束估计球位置 ===")

        # 步骤1-3: 得到相机坐标系中的射线方向
        distorted_image_point = self.pixel_to_image(pixel_point)
        ideal_image_point = self.remove_distortion(distorted_image_point)
        camera_ray = self.image_to_camera_ray(ideal_image_point)

        if verbose:
            print(f"像素坐标: ({pixel_point.x}, {pixel_point.y})")
            print(f"相机射线方向: ({camera_ray.x}, {camera_ray.y}, {camera_ray.z})")

        # 提取外参逆矩阵
        M = self.extrinsic_inv_.data

        # 求解lambda使得 P_world.z = 0
        A = M[2, 0] * camera_ray.x + M[2, 1] * camera_ray.y + M[2, 2] * camera_ray.z
        B = M[2, 3]

        if abs(A) < 1e-10:
            print("错误：射线与地面平行，无交点")
            return Point3D(0, 0, 0)

        lambda_val = -B / A

        if verbose:
            print(f"射线参数 lambda: {lambda_val}")

        # 计算交点
        camera_point = Point3D(
            camera_ray.x * lambda_val,
            camera_ray.y * lambda_val,
            camera_ray.z * lambda_val
        )
        world_point = self.camera_to_world(camera_point)

        if verbose:
            print(f"相机坐标系交点: ({camera_point.x}, {camera_point.y}, {camera_point.z})")
            print(f"世界坐标系球位置: ({world_point.x}, {world_point.y}, {world_point.z})")

        return world_point


class SimpleFootballDetector:
    def __init__(self, model_path: str, confidence_threshold=0.25, nms_threshold=0.4):
        self.model_path = model_path
        self.confidence_ = confidence_threshold
        self.nms_threshold_ = nms_threshold

        print(f"正在加载ONNX模型: {model_path}")

        # 初始化ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")

    def inference(self, img: np.ndarray) -> Optional[DetectionResult]:
        # 预处理图像
        input_img = self.preprocess(img)

        # 运行推理
        print("正在进行推理...")
        outputs = self.session.run([self.output_name], {self.input_name: input_img})
        print(f"推理完成! 输出形状: {outputs[0].shape}")

        # 后处理 - 只返回足球检测结果
        detection = self.postprocess(outputs[0], img.shape)
        return detection

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # 调整大小并归一化
        input_size = (640, 640)  # YOLOv8默认输入尺寸
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0

        # 转换通道顺序: HWC to NCHW
        input_img = np.transpose(normalized, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        return input_img

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> Optional[DetectionResult]:
        orig_h, orig_w = orig_shape[:2]
        input_size = 640
        output_data = outputs[0]

        football_detections = []

        # 遍历所有检测框，只收集足球检测结果（类别0）
        for i in range(output_data.shape[1]):
            detection = output_data[:, i]
            conf = detection[4]  # 置信度

            if conf > self.confidence_:
                x_center, y_center, w, h = detection[0:4]
                class_probs = detection[5:]
                class_id = np.argmax(class_probs)

                # 只处理足球类别（类别0）
                if class_id == 0:
                    # 坐标转换
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
                    u = int(x_center_orig)  # 水平中心
                    v = int(y1 + h_int)  # 底部

                    football_detections.append({
                        'position': (u, v),
                        'confidence': conf,
                        'bbox': (x1, y1, w_int, h_int)
                    })

        # 如果没有检测到足球，返回None
        if len(football_detections) == 0:
            print("未检测到足球")
            return None

        # 选择置信度最高的足球检测结果
        best_detection = max(football_detections, key=lambda x: x['confidence'])

        result = DetectionResult()
        result.position = best_detection['position']
        result.confidence = best_detection['confidence']
        result.bbox = best_detection['bbox']

        print(f"足球检测结果: 位置{result.position}, 置信度{result.confidence:.3f}")
        return result

    def draw_detection(self, image: np.ndarray, detection: DetectionResult,
                       world_position: Tuple[float, float, float]) -> np.ndarray:
        result_image = image.copy()

        if detection is None:
            print("没有检测结果可绘制")
            return result_image

        # 绘制检测框
        x, y, w, h = detection.bbox
        cv2.rectangle(result_image,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)  # 绿色框

        # 绘制底部中点
        u, v = detection.position
        cv2.circle(result_image, (u, v), 5, (0, 0, 255), -1)  # 红色圆点

        # 构建显示文本 - 包含世界坐标
        world_x, world_y, world_z = world_position
        display_text = f"Football: {detection.confidence:.2f}"
        position_text = f"World: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"

        # 绘制文本
        font_scale = 0.6
        text_color = (0, 0, 255)
        thickness = 2

        # 第一行文本：检测置信度
        text_size1 = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_org1 = (x, y - 20 if y - 20 > 20 else y + 20)
        cv2.putText(result_image, display_text, text_org1,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # 第二行文本：世界坐标
        text_size2 = cv2.getTextSize(position_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_org2 = (x, text_org1[1] + text_size1[1] + 5)
        cv2.putText(result_image, position_text, text_org2,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        print(f"绘制检测结果: 像素位置({u},{v}), 世界位置({world_x:.2f},{world_y:.2f},{world_z:.2f})")

        return result_image


# 工具函数
def create_rotation_x(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[1, 1] = cos_a
    rot.data[1, 2] = sin_a
    rot.data[2, 1] = -sin_a
    rot.data[2, 2] = cos_a

    return rot


def create_rotation_y(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[0, 0] = cos_a
    rot.data[0, 2] = -sin_a
    rot.data[2, 0] = sin_a
    rot.data[2, 2] = cos_a

    return rot


def create_rotation_z(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[0, 0] = cos_a
    rot.data[0, 1] = sin_a
    rot.data[1, 0] = -sin_a
    rot.data[1, 1] = cos_a

    return rot


def create_translation(tx: float, ty: float, tz: float) -> TransformMatrix:
    trans = TransformMatrix()
    trans.data[0, 3] = tx
    trans.data[1, 3] = ty
    trans.data[2, 3] = tz
    return trans


def create_extrinsic_matrix(pos_x: float, pos_y: float, pos_z: float,
                            pitch_deg: float, yaw_deg: float, roll_deg: float) -> TransformMatrix:
    rotation_z = create_rotation_z(yaw_deg)
    rotation_y = create_rotation_y(pitch_deg)
    rotation_x = create_rotation_x(roll_deg)

    rotation = rotation_x @ rotation_y @ rotation_z
    translation = create_translation(-pos_x, -pos_y, -pos_z)

    return rotation @ translation


def test_football_detection_with_3d(image_path: str, model_path: str, confidence_threshold=0.1):
    print("=== 足球检测与3D定位测试 ===")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功! 尺寸: {image.shape[1]} × {image.shape[0]}")

    try:
        # 初始化检测器
        detector = SimpleFootballDetector(model_path, confidence_threshold=confidence_threshold)

        # 进行检测
        detection = detector.inference(image)

        if detection is not None:
            u, v = detection.position
            print(f"成功检测到足球!")
            print(f"像素位置: ({u}, {v})")
            print(f"置信度: {detection.confidence:.3f}")
            print(f"边界框: {detection.bbox}")

            # 设置相机参数
            intrinsics = CameraIntrinsics()
            intrinsics.fx = 645.060547
            intrinsics.fy = 644.257935
            intrinsics.cx = 649.562866
            intrinsics.cy = 373.498932

            # 设置相机外参（位置和姿态）
            camera_x = 6.0  # 相机在世界坐标系中的X位置
            camera_y = -4.5  # 相机在世界坐标系中的Y位置
            camera_z = 1.0  # 相机在世界坐标系中的Z位置（高度）
            yaw_deg = -15.0  # 绕Z轴旋转（偏航角）
            pitch_deg = 0.0  # 绕Y轴旋转（俯仰角）
            roll_deg = -115.0  # 绕X轴旋转（滚转角）

            extrinsic = create_extrinsic_matrix(camera_x, camera_y, camera_z,
                                                pitch_deg, yaw_deg, roll_deg)

            # 创建反投影器
            back_projector = BackProjector(intrinsics, extrinsic)

            # 计算足球的世界坐标
            pixel_point = Point2D(u, v)
            world_position = back_projector.estimate_ball_position(pixel_point, verbose=True)

            # 更新检测结果的世界坐标
            detection.world_position = (world_position.x, world_position.y, world_position.z)

            print("\n=== 最终结果 ===")
            print(f"足球像素坐标: ({u}, {v})")
            print(f"足球世界坐标: ({world_position.x:.3f}, {world_position.y:.3f}, {world_position.z:.3f})")

            # 绘制检测结果（包含世界坐标）
            result_image = detector.draw_detection(image, detection, detection.world_position)

        else:
            print("未检测到足球")
            result_image = image.copy()

        # 显示结果
        cv2.imshow("Football Detection with 3D Position", result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = "football_detection_3d_result.jpg"
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

    # 进行检测并计算3D位置
    test_football_detection_with_3d(image_path, model_path, confidence_threshold=0.1)