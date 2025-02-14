#include "random_utils.h"

std::random_device rd1;
static std::mt19937 generator(rd1());
static std::uniform_real_distribution<double> distribution(0.0, 1.0);
static bool initialized = false;
std::normal_distribution<double> dist(0.0, 1.0);
std::random_device rd;  // Источник энтропии
std::mt19937 gen(rd());

// Инициализация генератора случайных чисел
void init_random() {
    if (!initialized) {
        generator.seed(static_cast<unsigned int>(std::time(nullptr)));
        initialized = true;
    }
}

// Генерация случайного числа в диапазоне [0.0, 1.0]
double rand_num() {
    return distribution(generator);
}

// Генерация нормально распределенного числа (метод Бокса-Мюллера)
double randn() {
    init_random();
    double val = std::clamp(dist(gen), 0.0, 1.0);
    return val;
}


// Генерация значения rho на основе параметра alpha
double new_rho(double alpha) {
    return (2.0 * rand_num() * alpha) - alpha;
}

// Генерация четырёх уникальных индексов
void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind) {
    int cnt = 0;
    std::array<bool, POP_SIZE> used_indices = {false};
    used_indices[cur_ind] = true;
    used_indices[best_ind] = true;

    while (cnt < 4) {
        size_t temp = gen_random_index();
        if ((!used_indices[temp])) {
            indexes[cnt] = temp;
            cnt++;
            used_indices[temp] = true;
        }
    }
}

// Генерация одного случайного индекса без проверки
size_t gen_random_index() {
    return generator() % POP_SIZE;  // Генерация случайного индекса
}

// Генерация случайного числа в диапазоне от -1 до 1 включительно
double rand_neg_one_to_one() {
    return 2.0 * rand_num() - 1.0;  // Генерация числа от -1 до 1
}

// Генерация случайного значения 0 или 1 типа unsigned char
unsigned char rand_binary() {
    // Используем rand_num() для генерации случайного числа в диапазоне [0.0, 1.0]
    double random_value = rand_num();

    // Если random_value < 0.5, возвращаем 0, иначе 1
    return (random_value < 0.5) ? 0 : 1;
}