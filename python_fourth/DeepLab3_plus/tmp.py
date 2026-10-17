import paddle.fluid as fluid
import numpy as np

# assign示例
# data = fluid.layers.fill_constant(shape=[3, 2],
#                                   value=2.5,
#                                   dtype='float64')  # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
# result1 = fluid.layers.create_tensor(dtype='float64')
#
# fluid.layers.assign(data, result1)  # result1 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
# result2 = fluid.layers.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
# result3 = fluid.layers.assign(
#     np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32'))  # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
#
# place = fluid.CPUPlace()
# exe = fluid.Executor(place)
#
# ret = exe.run(fetch_list=[result1, result2, result3])
# for r in ret:
#     print(r)
#     print("")

# transpose示例
# 请使用 append_batch_size=False 来避免
# 在数据张量中添加多余的batch大小维度
"""
x = [[[ 1  2  3  4] 
      [ 5  6  7  8] 
      [ 9 10 11 12]]
     [[13 14 15 16] 
      [17 18 19 20] 
      [21 22 23 24]]]
shape(x) =  [2,3,4]

# 例0
perm0 = [1,0,2]
y_perm0 = [[[ 1  2  3  4] 
            [13 14 15 16]]
           [[ 5  6  7  8]  
            [17 18 19 20]]
           [[ 9 10 11 12]  
            [21 22 23 24]]]
shape(y_perm0) = [3,2,4]
"""

"""
import paddle.fluid as fluid

x = fluid.layers.data(name='x',
                      shape=[2, 3, 4],
                      dtype='float32',
                      append_batch_size=False)
x_transposed = fluid.layers.transpose(x, perm=[1, 0, 2])
print(x_transposed.shape)
# (3L, 2L, 4L)
"""

import paddle.fluid as fluid
import numpy as np

in1 = np.array([[[[5, 3, 2],
                  [6, 8, 1]],
                 [[5, 2, 4],
                  [4, 7, 7]]],
                [[[1, 3, 4],
                  [2, 2, 1]],
                 [[2, 3, 4],
                  [4, 4, 5]]]])
with fluid.dygraph.guard():
    x = fluid.dygraph.to_variable(in1)
    # out1 = fluid.layers.argmax(x=x, axis=-1)
    # print(out1.numpy())

    # out2 = fluid.layers.argmax(x=x, axis=0)
    # print(out2.numpy())

    out3 = fluid.layers.argmax(x=x, axis=1)
    print(out3.numpy())

    # out4 = fluid.layers.argmax(x=x, axis=2)
    # print(out4.numpy())
