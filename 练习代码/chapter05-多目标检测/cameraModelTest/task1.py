import cv2
import numpy as np
from typing import Tuple
import math


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

    def world_to_camera(self, world_point: Point3D) -> Point3D:
        """将世界坐标系中的点转换到相机坐标系"""
        world_homogeneous = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        camera_homogeneous = self.extrinsic_.data @ world_homogeneous

        if abs(camera_homogeneous[3]) > 1e-10:
            return Point3D(
                camera_homogeneous[0] / camera_homogeneous[3],
                camera_homogeneous[1] / camera_homogeneous[3],
                camera_homogeneous[2] / camera_homogeneous[3]
            )
        else:
            return Point3D(
                camera_homogeneous[0],
                camera_homogeneous[1],
                camera_homogeneous[2]
            )

    def apply_distortion(self, ideal_point: Point2D) -> Point2D:
        """应用畸变模型（正向）"""
        x = ideal_point.x / self.intrinsics_.fx
        y = ideal_point.y / self.intrinsics_.fy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = 1.0 + self.intrinsics_.k1 * r2 + self.intrinsics_.k2 * r4 + self.intrinsics_.k3 * r6
        delta_x = 2 * self.intrinsics_.p1 * x * y + self.intrinsics_.p2 * (r2 + 2 * x * x)
        delta_y = self.intrinsics_.p1 * (r2 + 2 * y * y) + 2 * self.intrinsics_.p2 * x * y

        x_distorted = x * radial_factor + delta_x
        y_distorted = y * radial_factor + delta_y

        x_distorted_pixel = x_distorted * self.intrinsics_.fx
        y_distorted_pixel = y_distorted * self.intrinsics_.fy

        return Point2D(x_distorted_pixel, y_distorted_pixel)

    def camera_to_image(self, camera_point: Point3D) -> Point2D:
        """将相机坐标系中的点投影到图像平面"""
        if abs(camera_point.z) < 1e-10:
            return Point2D(0, 0)

        x_ideal = (camera_point.x / camera_point.z) * self.intrinsics_.fx
        y_ideal = (camera_point.y / camera_point.z) * self.intrinsics_.fy

        return Point2D(x_ideal, y_ideal)

    def image_to_pixel(self, image_point: Point2D) -> Point2D:
        """将图像坐标转换到像素坐标"""
        x = image_point.x + self.intrinsics_.cx
        y = image_point.y + self.intrinsics_.cy
        return Point2D(x, y)

    def world_to_pixel(self, world_point: Point3D, verbose=False) -> Point2D:
        """将世界坐标系中的点投影到像素坐标"""
        if verbose:
            print("=== 世界坐标到像素坐标投影 ===")
            print(f"世界坐标: ({world_point.x}, {world_point.y}, {world_point.z})")

        # 步骤1: 世界坐标 -> 相机坐标
        camera_point = self.world_to_camera(world_point)
        if verbose:
            print(f"相机坐标: ({camera_point.x}, {camera_point.y}, {camera_point.z})")

        # 检查点是否在相机前方
        if camera_point.z <= 0:
            print("警告：点在相机后方，无法投影")
            return Point2D(-1, -1)

        # 步骤2: 相机坐标 -> 理想图像坐标
        ideal_image_point = self.camera_to_image(camera_point)
        if verbose:
            print(f"理想图像坐标: ({ideal_image_point.x}, {ideal_image_point.y})")

        # 步骤3: 应用畸变 -> 畸变图像坐标
        distorted_image_point = self.apply_distortion(ideal_image_point)
        if verbose:
            print(f"畸变图像坐标: ({distorted_image_point.x}, {distorted_image_point.y})")

        # 步骤4: 图像坐标 -> 像素坐标
        pixel_point = self.image_to_pixel(distorted_image_point)
        if verbose:
            print(f"像素坐标: ({pixel_point.x}, {pixel_point.y})")

        return pixel_point


class WorldPointMarker:
    def __init__(self):
        pass

    def mark_world_point(self, image: np.ndarray, world_point: Point3D,
                         back_projector: BackProjector, label: str = "") -> np.ndarray:
        """在图像上标记世界坐标系中的点"""
        result_image = image.copy()

        # 将世界坐标投影到像素坐标
        pixel_point = back_projector.world_to_pixel(world_point, verbose=True)

        if pixel_point.x >= 0 and pixel_point.y >= 0:
            u = int(pixel_point.x)
            v = int(pixel_point.y)

            # 绘制红心标记
            heart_color = (0, 0, 255)  # 红色
            marker_size = 20

            # 绘制心形标记
            self.draw_heart(result_image, (u, v), marker_size, heart_color)

            # 添加坐标文本
            if label:
                text = f"{label}: ({world_point.x:.1f}, {world_point.y:.1f}, {world_point.z:.1f})"
            else:
                text = f"World: ({world_point.x:.1f}, {world_point.y:.1f}, {world_point.z:.1f})"

            cv2.putText(result_image, text, (u + marker_size + 5, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, heart_color, 2)

            print(f"世界坐标 ({world_point.x}, {world_point.y}, {world_point.z}) 投影到像素坐标 ({u}, {v})")
        else:
            print(f"世界坐标 ({world_point.x}, {world_point.y}, {world_point.z}) 无法投影到图像内")

        return result_image

    def draw_heart(self, image: np.ndarray, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
        """绘制心形标记"""
        x, y = center

        # 心形参数方程
        points = []
        for t in np.linspace(0, 2 * np.pi, 100):
            # 心形参数方程
            heart_x = 16 * (np.sin(t) ** 3)
            heart_y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

            # 缩放和移动
            px = int(x + heart_x * size / 16)
            py = int(y - heart_y * size / 16)  # 注意y轴方向

            points.append((px, py))

        # 绘制填充的心形
        if len(points) > 2:
            pts = np.array(points, np.int32)
            cv2.fillPoly(image, [pts], color)

        # 添加白色边框使心形更明显
        cv2.polylines(image, [np.array(points, np.int32)], True, (255, 255, 255), 2)


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


def test_world_point_projection(image_path: str):
    print("=== 世界坐标到像素坐标投影测试 ===")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功! 尺寸: {image.shape[1]} × {image.shape[0]}")

    try:
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

        # 创建标记器
        marker = WorldPointMarker()

        # 标记指定的世界坐标点 (7, -3, 0)
        target_world_point = Point3D(7, -3, 0)
        result_image = marker.mark_world_point(image, target_world_point, back_projector, "Target")

        # 显示结果
        cv2.imshow("World Point Projection", result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = "world_point_projection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请替换为实际的文件路径
    image_path = "football_image.jpg"

    # 进行世界坐标到像素坐标的投影测试
    test_world_point_projection(image_path)