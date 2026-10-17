'''
封装模型预测
'''
import paddle
import paddle.fluid as fluid
import numpy as np


class HousePricePredict:

    def __init__(self,model_path):
        self.model_path = model_path
        # 执行器
        place = fluid.CPUPlace()
        self.infer_exe = fluid.Executor(place=place)

        (self.infer_program,
         self.feed_name,
         self.target_name) = fluid.io.load_inference_model(dirname=model_path,
                                      executor=self.infer_exe)


    def predict(self,data):
        data = np.array(data).reshape(-1,13)  # n行13列   多少行都能接收  (16,) --> (-1,4)  == (4,4)
        # 执行预测
        res = self.infer_exe.run(program=self.infer_program,
                                 feed={self.feed_name[0]:data},
                                 fetch_list=self.target_name)
        return res[0]





if __name__ == '__main__':
    model = HousePricePredict('../model/boston/infer')
    pred_data = np.random.normal(1,1,size=(26 ,)).astype('float32')
    print(model.predict(pred_data))
