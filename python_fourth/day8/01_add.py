'''
执行两个张量的相加
'''
import paddle.fluid as fluid

# x = 100.0
# y = 200.0

# 定义 default_main_program

# "Default main program"（默认主程序）
# 通常是指在编程环境或项目中自动识别为程序入口点的主要执行文件
x = fluid.layers.fill_constant(shape=(1,),dtype='float64',value=100.0)
y = fluid.layers.fill_constant(shape=(1,),dtype='float64',value=200.0)

res = x + y

# print(res)


# 执行器
place = fluid.CPUPlace()  # 执行CPU设备
# fluid.CUDAPlace(0)   # 执行GPU设备
exe = fluid.Executor(place=place)

result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[res])

print(result[0])


# 静态图：定义与运行相分离
# 动态图：print(res)之后直接就输出答案

# shape 是描述数组（或张量）维度的元组。
# (1,) 是一个单元素的元组（注意逗号），表示：
# 这是一个 1维数组（因为元组里只有1个数字）。
# 这个维度的大小是 1（即数组里只有1个元素）。

# shape=()：标量（0维数组，比如 np.array(42)）。
# shape=(n,)：一维数组，包含 n 个元素（如 np.array([1, 2, 3]) 的 shape 是 (3,)）。
# shape=(1, n)：二维数组，1行n列（如 np.array([[1, 2, 3]]) 的 shape 是 (1, 3)）。