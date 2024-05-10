import pandas as pd
import numpy as np

chat_id = 387152568  # Ваш chat ID, не меняйте название переменной

from scipy.stats import t


def solution(x_success: int, x_cnt: int, y_success: int, y_cnt: int) -> bool:
    # Преобразуем данные в формат, удобный для использования
    control_data = np.concatenate([np.ones(x_success), np.zeros(x_cnt - x_success)])
    test_data = np.concatenate([np.ones(y_success), np.zeros(y_cnt - y_success)])

    alpha = 0.05
    power = 0.8

    # Средние значения для обеих групп
    control_mean = np.mean(control_data)
    test_mean = np.mean(test_data)

    # Стандартные отклонения для обеих групп
    control_std = np.std(control_data)
    test_std = np.std(test_data)

    # Стандартная ошибка разности между средними значениями
    stderr_diff = np.sqrt(
        (control_std**2 / len(control_data)) + (test_std**2 / len(test_data))
    )

    # Критическое значение t-статистики
    t_critical = t.ppf(1 - alpha / 2, df=len(control_data) + len(test_data) - 2)

    # Доверительный интервал для разности между средними значениями
    confidence_interval = (control_mean - test_mean) + np.array(
        [-1, 1]
    ) * t_critical * stderr_diff

    # Определяем, лежит ли доверительный интервал полностью выше или ниже нуля
    if confidence_interval[0] > 0 or confidence_interval[1] < 0:
        # Статистически значимо
        return True
    else:
        # Не является статистически значимым
        return False
