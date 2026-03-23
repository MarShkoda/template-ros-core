#!/usr/bin/env python3

import numpy as np
from gym_duckietown.envs import DuckietownEnv
import cv2
import math
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# Добавляем путь к модулю lane_diff_tracker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lane_diff_tracker import LaneDeviationTracker, track_lane_performance

# Константы для LQR
DT = 1/30  # шаг времени
V = 0.5    # скорость (м/с)

class LQRController:
    """LQR контроллер для управления движением в полосе"""
    
    def __init__(self):
        self.L = 0.1  # длина колесной базы (м) - для Duckietown
        
        # Матрицы состояния для линейной модели
        # Состояние: [e, e_dot, psi, psi_dot] 
        # e - латеральное отклонение, psi - угол курса
        
        # Матрица A (системная)
        self.A = np.array([
            [1, DT, V*DT, 0],
            [0, 1, 0, V*DT],
            [0, 0, 1, DT],
            [0, 0, 0, 1]
        ])
        
        # Матрица B (управляющая) - управление: угол поворота колес (delta)
        self.B = np.array([
            [0],
            [V*DT**2/(2*self.L)],
            [0],
            [V*DT/self.L]
        ])
        
        # Матрицы весов Q и R
        # Q - веса состояний, R - вес управления
        self.Q = np.diag([10.0, 1.0, 5.0, 1.0])  # Больше вес на отклонение и угол
        self.R = np.array([[0.1]])  # Меньше вес на управление = более агрессивный контроль
        
        # Вычисляем матрицу усиления LQR
        self.K = self._compute_lqr_gain()
        
        # Состояние контроллера
        self.state = np.zeros((4, 1))  # [e, e_dot, psi, psi_dot]
        self.prev_lateral = 0
        self.prev_heading = 0
        
        print(f"LQR контроллер инициализирован")
        print(f"Усиление K: {self.K.flatten()}")
        
    def _compute_lqr_gain(self):
        # дискретное алгебраическое уравнение Риккати
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            # матрица усиления
            K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
            return K
        except Exception as e:
            print(f"Ошибка вычисления LQR: {e}")
            # эмпирические коэффициенты
            return np.array([[-2.0, -1.0, -3.0, -0.5]])
    
    def update(self, lateral_error, heading_error):
        """
        Обновление LQR контроллера
        
        Args:
            lateral_error: латеральное отклонение (м)
            heading_error: ошибка угла курса (рад)
        
        Returns:
            control: управляющее воздействие (угол поворота колес)
        """
        # Вычисляем производные
        lateral_dot = (lateral_error - self.prev_lateral) / DT if self.prev_lateral != 0 else 0
        heading_dot = (heading_error - self.prev_heading) / DT if self.prev_heading != 0 else 0
        
        # Обновляем состояние
        self.state = np.array([
            [lateral_error],
            [lateral_dot],
            [heading_error],
            [heading_dot]
        ])
        
        # Вычисляем управление
        control = -self.K @ self.state
        
        # Сохраняем предыдущие значения
        self.prev_lateral = lateral_error
        self.prev_heading = heading_error
        
        # Ограничиваем управление
        control = np.clip(control, -1.5, 1.5)
        
        return float(control)
    
    def reset(self):
        """Сброс состояния контроллера"""
        self.state = np.zeros((4, 1))
        self.prev_lateral = 0
        self.prev_heading = 0

class PIDController:
    """PID контроллер для сравнения"""
    
    def __init__(self):
        # Коэффициенты PID (подобраны эмпирически)
        self.Kp = 0.05   # Пропорциональный
        self.Ki = 0.01   # Интегральный
        self.Kd = 0.02   # Дифференциальный
        
        self.prev_error = 0
        self.integral = 0
        self.dt = DT
        
        print(f"PID контроллер инициализирован")
        print(f"Коэффициенты: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
    
    def update(self, lateral_error, heading_error):
        """
        Обновление PID контроллера
        
        Args:
            lateral_error: латеральное отклонение (м) - основной сигнал
            heading_error: ошибка угла курса (рад) - для коррекции
        
        Returns:
            control: управляющее воздействие
        """
        # Основная ошибка - латеральное отклонение
        error = lateral_error * 100  # Масштабируем для лучшей чувствительности
        
        # Добавляем компонент угла
        error += heading_error * 50
        
        # Пропорциональная составляющая
        P = self.Kp * error
        
        # Интегральная составляющая (с насыщением)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -100, 100)  # Анти-windup
        I = self.Ki * self.integral
        
        # Дифференциальная составляющая
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        
        # Суммируем
        control = P + I + D
        
        # Сохраняем ошибку
        self.prev_error = error
        
        # Ограничиваем управление
        control = np.clip(control, -1.5, 1.5)
        
        return control
    
    def reset(self):
        """Сброс состояния контроллера"""
        self.prev_error = 0
        self.integral = 0

def calc_err(obs):
    """Детекция линий для вычисления ошибки"""
    img = np.ascontiguousarray(obs)

    

    return   # Ошибка в пикселях

def bird(source):
    """Bird-eye view трансформация"""
    PI = 3.1415926

    frameWidth = 640
    frameHeight = 480

    alpha = (17 - 90) * PI / 180
    beta = (90 - 90) * PI / 180
    gamma = (90 - 90) * PI / 180
    focalLength = 688
    dist = 477

    image_size = (frameWidth, frameHeight)
    w, h = image_size

    A1 = np.array([[1, 0, -w / 2],
                [0, 1, -h / 2],
                [0, 0, 0],
                [0, 0, 1]], dtype=np.float32)

    RX = np.array([[1, 0, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha), 0],
                [0, math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                [0, 1, 0, 0],
                [math.sin(beta), 0, math.cos(beta), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                [math.sin(gamma), math.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    R = np.dot(np.dot(RX, RY), RZ)

    T = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dist],
                [0, 0, 0, 1]], dtype=np.float32)

    K = np.array([[focalLength, 0, w / 2, 0],
                [0, focalLength, h / 2, 0],
                [0, 0, 1, 0]], dtype=np.float32)

    transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)
    
    destination = cv2.warpPerspective(source, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    
    return destination

from image_processing.line_processing import ImageLineProcessing, ColorLine

PIXEL_TO_METER = 0.0012  # ≈ 1.2 мм / пиксель (реалистично для Duckietown)

def compute_lateral_and_heading_error(obs):
    """
    Возвращает:
        lateral_error (м)
        heading_error (рад)
    """
    h, w, _ = obs.shape
    y_ref = int(h * 0.75)

    # --- обработка линий ---
    yellow_proc = ImageLineProcessing(obs, w, h, ColorLine.yellow)
    white_proc  = ImageLineProcessing(obs, w, h, ColorLine.white)

    angle_y, line_y = yellow_proc.process()
    angle_w, line_w = white_proc.process()

    if line_y is None or line_w is None:
        return 0.0, 0.0  # fallback

    # --- расширяем линии ---
    ly = yellow_proc.get_extended_line(line_y)
    lw = white_proc.get_extended_line(line_w)

    def x_at_y(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return x1
        return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    x_left  = x_at_y(ly, y_ref)
    x_right = x_at_y(lw, y_ref)

    lane_center_x = (x_left + x_right) / 2
    image_center_x = w / 2

    lateral_error = (lane_center_x - image_center_x) * PIXEL_TO_METER

    # --- угол ---
    lane_angle = np.nanmean([angle_y, angle_w])
    heading_error = lane_angle - np.pi / 2

    return lateral_error, heading_error

def run_controller_simulation(controller_type='pid', output_dir="simulation_output"):
    """Запуск симуляции с заданным контроллером"""
    MAX_STEPS = 1000
    # Инициализация среды
    env = DuckietownEnv(
        **{"seed": 128546,
        "map_name": "straight_road",
        "max_steps": MAX_STEPS,  # Уменьшено для быстрого сравнения
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 30,
        "full_transparency": True,
        "distortion": True,
        "domain_rand": False
        }
    )
    
    # Инициализация трекера
    lane_tracker = LaneDeviationTracker(env, max_steps=MAX_STEPS, real_time_display=False)
    
    # Инициализация контроллера
    if controller_type.lower() == 'lqr':
        controller = LQRController()
        controller_name = "LQR"
    else:
        controller = PIDController()
        controller_name = "PID"
    
    print(f"\n{'='*60}")
    print(f"Запуск симуляции с {controller_name} контроллером")
    print(f"{'='*60}")
    
    done = False
    obs = env.reset()
    step = 0
    
    # История управления для анализа
    control_history = []
    lateral_history = []
    heading_history = []
    
    while not done and step < MAX_STEPS:
        lateral, heading = track_lane_performance(env, lane_tracker, step)
        pixel_error = calc_err(obs)
        lateral_error_m = pixel_error * 0.001  # Примерное преобразование

        

        if controller_type.lower() == 'pid':
            control = controller.update(lateral, heading)
        else:
            # Для LQR используем только ошибки из трекера
            control = controller.update(lateral, heading)
        
        # Сохраняем историю
        lateral_history.append(lateral)
        heading_history.append(heading)
        control_history.append(control)
        
        # Формирование действия
        action = [0.5, control]
        
        # Выполнение шага
        obs, rew, done, info = env.step(np.array(action))
        
        # Прогресс
        if step % 100 == 0:
            print(f"Шаг {step}: отклонение = {lateral:.4f} м, угол = {np.degrees(heading):.2f}°, управление = {control:.3f}")
        
        step += 1
    
    # Закрытие среды
    env.close()
    
    # Создаем поддиректорию для контроллера
    controller_dir = os.path.join(output_dir, controller_name)
    if not os.path.exists(controller_dir):
        os.makedirs(controller_dir)
    
    # Сохраняем результаты
    results = {
        'controller': controller_name,
        'steps': lane_tracker.steps,
        'lateral_deviations': lane_tracker.lateral_deviations,
        'heading_angles': lane_tracker.heading_angles,
        'control_signals': control_history,
        'lateral_history': lateral_history,
        'heading_history': heading_history,
        'stats': {
            'mean_lateral': lane_tracker.mean_lateral,
            'std_lateral': lane_tracker.std_lateral,
            'mean_heading': lane_tracker.mean_heading,
            'std_heading': lane_tracker.std_heading,
            'max_deviation': max(lane_tracker.lateral_deviations, key=abs),
            'within_005': np.mean(np.abs(lane_tracker.lateral_deviations) < 0.05) * 100
        }
    }
    
    # Сохраняем статистику
    stats_file = os.path.join(controller_dir, f"{controller_name}_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"СТАТИСТИКА {controller_name} КОНТРОЛЛЕРА\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Всего шагов: {len(lane_tracker.steps)}\n")
        f.write(f"Среднее латеральное отклонение: {lane_tracker.mean_lateral:.6f} м\n")
        f.write(f"Станд артное отклонение: {lane_tracker.std_lateral:.6f} м\n")
        f.write(f"Средний угол: {np.degrees(lane_tracker.mean_heading):.4f}°\n")
        f.write(f"Стандартное отклонение угла: {np.degrees(lane_tracker.std_heading):.4f}°\n")
        f.write(f"Максимальное отклонение: {max(lane_tracker.lateral_deviations, key=abs):.6f} м\n")
        f.write(f"RMS отклонение: {np.sqrt(np.mean(np.array(lane_tracker.lateral_deviations)**2)):.6f} м\n")
        f.write(f"Доля времени в пределах ±0.05 м: {np.mean(np.abs(lane_tracker.lateral_deviations) < 0.05)*100:.1f}%\n")
    
    print(f"Симуляция с {controller_name} завершена")
    print(f"Результаты сохранены в {controller_dir}")
    
    return results

def plot_controller_comparison(pid_results, lqr_results, output_dir="simulation_output"):
    """Создание графиков сравнения контроллеров"""
    
    # Создаем директорию для сравнения
    comparison_dir = os.path.join(output_dir, "comparison")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 1. График латерального отклонения (общий)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # PID
    ax1.plot(pid_results['steps'], pid_results['lateral_deviations'], 
             'b-', linewidth=1.5, alpha=0.8, label=f"PID (ср.={pid_results['stats']['mean_lateral']:.4f} м)")
    
    # LQR
    ax1.plot(lqr_results['steps'], lqr_results['lateral_deviations'], 
             'r-', linewidth=1.5, alpha=0.8, label=f"LQR (ср.={lqr_results['stats']['mean_lateral']:.4f} м)")
    
    # Горизонтальные линии
    ax1.axhline(y=0, color='g', linestyle='-', linewidth=1, label='Центр полосы')
    ax1.axhline(y=0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axhline(y=-0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Зона ±0.05 м
    ax1.fill_between([min(pid_results['steps']), max(pid_results['steps'])], 
                     -0.05, 0.05, alpha=0.1, color='green', label='Зона ±0.05 м')
    
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Латеральное отклонение (м)')
    ax1.set_title('Сравнение латерального отклонения: PID vs LQR')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Сохраняем
    fig1.savefig(os.path.join(comparison_dir, "lateral_deviation_comparison.png"), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. График угла (общий)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # PID
    ax2.plot(pid_results['steps'], np.degrees(pid_results['heading_angles']), 
             'b-', linewidth=1.5, alpha=0.8, label=f"PID (ср.={np.degrees(pid_results['stats']['mean_heading']):.1f}°)")
    
    # LQR
    ax2.plot(lqr_results['steps'], np.degrees(lqr_results['heading_angles']), 
             'r-', linewidth=1.5, alpha=0.8, label=f"LQR (ср.={np.degrees(lqr_results['stats']['mean_heading']):.1f}°)")
    
    ax2.axhline(y=0, color='g', linestyle='-', linewidth=1, label='Идеальный угол')
    
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Угол относительно полосы (градусы)')
    ax2.set_title('Сравнение угла: PID vs LQR')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig2.savefig(os.path.join(comparison_dir, "heading_angle_comparison.png"), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. График управления (сигналы управления)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    ax3.plot(pid_results['steps'][:len(pid_results['control_signals'])], 
             pid_results['control_signals'], 
             'b-', linewidth=1.5, alpha=0.8, label='PID управление')
    
    ax3.plot(lqr_results['steps'][:len(lqr_results['control_signals'])], 
             lqr_results['control_signals'], 
             'r-', linewidth=1.5, alpha=0.8, label='LQR управление')
    
    ax3.axhline(y=0, color='g', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax3.set_xlabel('Шаг')
    ax3.set_ylabel('Управляющий сигнал')
    ax3.set_title('Сравнение управляющих сигналов: PID vs LQR')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    fig3.savefig(os.path.join(comparison_dir, "control_signals_comparison.png"), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Отдельные графики для каждого контроллера
    
    # PID отдельно
    fig_pid, (ax_pid1, ax_pid2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # PID латеральное отклонение
    ax_pid1.plot(pid_results['steps'], pid_results['lateral_deviations'], 
                 'b-', linewidth=1.5, alpha=0.8)
    ax_pid1.axhline(y=0, color='g', linestyle='-', linewidth=1)
    ax_pid1.axhline(y=0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_pid1.axhline(y=-0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_pid1.fill_between([min(pid_results['steps']), max(pid_results['steps'])], 
                         -0.05, 0.05, alpha=0.1, color='green')
    ax_pid1.set_ylabel('Отклонение (м)')
    ax_pid1.set_title(f'PID контроллер: Латеральное отклонение (ср.={pid_results["stats"]["mean_lateral"]:.4f} м)')
    ax_pid1.grid(True, alpha=0.3)
    
    # PID угол
    ax_pid2.plot(pid_results['steps'], np.degrees(pid_results['heading_angles']), 
                 'g-', linewidth=1.5, alpha=0.8)
    ax_pid2.axhline(y=0, color='g', linestyle='-', linewidth=1)
    ax_pid2.set_xlabel('Шаг')
    ax_pid2.set_ylabel('Угол (градусы)')
    ax_pid2.set_title(f'PID контроллер: Угол (ср.={np.degrees(pid_results["stats"]["mean_heading"]):.1f}°)')
    ax_pid2.grid(True, alpha=0.3)
    
    fig_pid.tight_layout()
    fig_pid.savefig(os.path.join(comparison_dir, "PID_controller_results.png"), 
                    dpi=150, bbox_inches='tight')
    plt.close(fig_pid)
    
    # LQR отдельно
    fig_lqr, (ax_lqr1, ax_lqr2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # LQR латеральное отклонение
    ax_lqr1.plot(lqr_results['steps'], lqr_results['lateral_deviations'], 
                 'r-', linewidth=1.5, alpha=0.8)
    ax_lqr1.axhline(y=0, color='g', linestyle='-', linewidth=1)
    ax_lqr1.axhline(y=0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_lqr1.axhline(y=-0.05, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_lqr1.fill_between([min(lqr_results['steps']), max(lqr_results['steps'])], 
                         -0.05, 0.05, alpha=0.1, color='green')
    ax_lqr1.set_ylabel('Отклонение (м)')
    ax_lqr1.set_title(f'LQR контроллер: Латеральное отклонение (ср.={lqr_results["stats"]["mean_lateral"]:.4f} м)')
    ax_lqr1.grid(True, alpha=0.3)
    
    # LQR угол
    ax_lqr2.plot(lqr_results['steps'], np.degrees(lqr_results['heading_angles']), 
                 'orange', linewidth=1.5, alpha=0.8)
    ax_lqr2.axhline(y=0, color='g', linestyle='-', linewidth=1)
    ax_lqr2.set_xlabel('Шаг')
    ax_lqr2.set_ylabel('Угол (градусы)')
    ax_lqr2.set_title(f'LQR контроллер: Угол (ср.={np.degrees(lqr_results["stats"]["mean_heading"]):.1f}°)')
    ax_lqr2.grid(True, alpha=0.3)
    
    fig_lqr.tight_layout()
    fig_lqr.savefig(os.path.join(comparison_dir, "LQR_controller_results.png"), 
                    dpi=150, bbox_inches='tight')
    plt.close(fig_lqr)
    
    # 5. Сводная таблица сравнения
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    # Создаем таблицу
    summary_data = [
        ['Параметр', 'PID контроллер', 'LQR контроллер', 'Лучший'],
        ['Среднее отклонение (м)', 
         f"{pid_results['stats']['mean_lateral']:.6f}", 
         f"{lqr_results['stats']['mean_lateral']:.6f}",
         'PID' if abs(pid_results['stats']['mean_lateral']) < abs(lqr_results['stats']['mean_lateral']) else 'LQR'],
        
        ['Стандартное отклонение (м)', 
         f"{pid_results['stats']['std_lateral']:.6f}", 
         f"{lqr_results['stats']['std_lateral']:.6f}",
         'PID' if pid_results['stats']['std_lateral'] < lqr_results['stats']['std_lateral'] else 'LQR'],
        
        ['Максимальное отклонение (м)', 
         f"{abs(pid_results['stats']['max_deviation']):.6f}", 
         f"{abs(lqr_results['stats']['max_deviation']):.6f}",
         'PID' if abs(pid_results['stats']['max_deviation']) < abs(lqr_results['stats']['max_deviation']) else 'LQR'],
        
        ['Время в зоне ±0.05 м (%)', 
         f"{pid_results['stats']['within_005']:.1f}", 
         f"{lqr_results['stats']['within_005']:.1f}",
         'PID' if pid_results['stats']['within_005'] > lqr_results['stats']['within_005'] else 'LQR'],
        
        ['Средний угол (°)', 
         f"{np.degrees(pid_results['stats']['mean_heading']):.2f}", 
         f"{np.degrees(lqr_results['stats']['mean_heading']):.2f}",
         'PID' if abs(np.degrees(pid_results['stats']['mean_heading'])) < abs(np.degrees(lqr_results['stats']['mean_heading'])) else 'LQR']
    ]
    
    table = ax_summary.table(cellText=summary_data, 
                             cellLoc='center', 
                             loc='center',
                             colWidths=[0.3, 0.23, 0.23, 0.24])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Выделение заголовков
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Выделение лучших результатов
    for i in range(1, len(summary_data)):
        best_col = summary_data[i][3]
        if best_col == 'PID':
            table[(i, 1)].set_facecolor('#d4edda')  # Зеленый для PID
        elif best_col == 'LQR':
            table[(i, 2)].set_facecolor('#f8d7da')  # Красный для LQR
    
    ax_summary.set_title('Сравнение производительности контроллеров', fontsize=14, pad=20)
    
    fig_summary.savefig(os.path.join(comparison_dir, "performance_summary.png"), 
                       dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    
    print(f"\nГрафики сравнения сохранены в {comparison_dir}")
    
    # Вывод сводки в консоль
    print(f"\n{'='*60}")
    print("СВОДКА СРАВНЕНИЯ КОНТРОЛЛЕРОВ")
    print(f"{'='*60}")
    print(f"{'Параметр':<30} {'PID':<15} {'LQR':<15} {'Лучший':<10}")
    print(f"{'-'*70}")
    
    for i in range(1, len(summary_data)):
        param = summary_data[i][0]
        pid_val = summary_data[i][1]
        lqr_val = summary_data[i][2]
        best = summary_data[i][3]
        
        print(f"{param:<30} {pid_val:<15} {lqr_val:<15} {best:<10}")
    
    print(f"{'='*60}")

def run_comparison_study():
    """Основная функция для сравнения контроллеров"""
    
    # Создаем основную директорию
    output_dir = "controller_comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"{'='*60}")
    print("ЗАПУСК СРАВНИТЕЛЬНОГО ИССЛЕДОВАНИЯ КОНТРОЛЛЕРОВ")
    print(f"{'='*60}")
    
    # Запускаем симуляцию с PID контроллером
    print("\n1. Запуск симуляции с PID контроллером...")
    pid_results = run_controller_simulation('pid', output_dir)
    
    # Запускаем симуляцию с LQR контроллером
    print("\n2. Запуск симуляции с LQR контроллером...")
    lqr_results = run_controller_simulation('lqr', output_dir)
    
    # Создаем графики сравнения
    print("\n3. Создание графиков сравнения...")
    plot_controller_comparison(pid_results, lqr_results, output_dir)
    
    print(f"\n{'='*60}")
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Все результаты сохранены в директории: {output_dir}/")
    print(f"\nСодержание:")
    print(f"  {output_dir}/PID/ - результаты PID контроллера")
    print(f"  {output_dir}/LQR/ - результаты LQR контроллера")
    print(f"  {output_dir}/comparison/ - графики сравнения")
    
    return pid_results, lqr_results

if __name__ == "__main__":
    run_comparison_study()