#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // Информация об авторе
    cout << "Автор: Иванов Иван Иванович" << endl;
    cout << "Группа: ИВТ-123" << endl << endl;

    int N;
    cin >> N;

    // Частный случай
    if (N == 1) {
        cout << "1" << endl;
        return 0;
    }

    // Если N делится на 2 или 5, числа из единиц не делятся
    if (N % 2 == 0 || N % 5 == 0) {
        cout << "no" << endl;
        return 0;
    }

    vector<bool> visited(N, false);
    int remainder = 1;
    int length = 1;

    while (remainder != 0 && !visited[remainder]) {
        visited[remainder] = true;
        remainder = (remainder * 10 + 1) % N;
        length++;
    }

    if (remainder == 0) {
        // Выводим length единиц
        cout << string(length, '1') << endl;
    } else {
        cout << "no" << endl;
    }

    return 0;
}
