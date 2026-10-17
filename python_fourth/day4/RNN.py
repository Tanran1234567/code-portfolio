import numpy as np
w_x = 0.5
w_h = 0.8
b = 0.1
# 定义输入数据
x = [1.0,2.0,3.0]
h = 0.0

# 循环计算
hidden_state = []


for x_t in x:
    y = x_t * w_x + h * w_h + b
    h = np.tanh(y)
    hidden_state.append(h)

print(hidden_state)