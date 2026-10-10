import { defineNavbarConfig } from 'vuepress-theme-plume'

export default defineNavbarConfig([
  { text: '首页', link: '/' },
  {
    text: '基础学习',
    items: [
      { text: '机器学习', link: '/MachineLearning/MachineLearning/01_introduction' },
      { text: '数据锻造坊', link: '/DataFoundry/DataStructure/01_xxx' },
      { text: '每日一题', link: '/CodeExam/AI/Introduction' },
    ],
  },
  {
    text: '深度学习',
    items: [
      { text: 'PyTorch', link: '/DeepLearning/PyTorch/' },
      { text: '跨时代算法', link: '/DeepLearning/EpochalAlgorithms/01_ImageClassification' },
      { text: '网络结构', link: '/DeepLearning/NetworkStructure/01_MLP' },
      { text: '激活函数', link: '/DeepLearning/ActivationFunction/01_Sigmoid' },
      { text: '损失函数', link: '/DeepLearning/LossFunction/01_MSE' },
      { text: '优化器', link: '/DeepLearning/Optimizer/01_SGD' },
      { text: '归一化', link: '/DeepLearning/Normalization/01_BatchNorm' },
      { text: '评价指标', link: '/DeepLearning/EvaluationMetrics/01_CommonMetrics' },
    ],
  },
  {
    text: '计算机视觉',
    items: [
      { text: '图像处理', link: '/ComputerVision/ImageProcessing/OpenCV1/01_综述' },
      { text: '图像生成', link: '/ComputerVision/ImageGeneration/01_综述' },
      { text: '目标检测', link: '/ComputerVision/TargetDetection/01_综述' },
      { text: 'YOLO 系列', link: '/ComputerVision/YOLOSeries/01_YOLOv8_Detection' },
    ],
  },
  {
    text: '更多',
    items: [
      { text: '桌面应用', link: '/desktop_app/' },
      { text: '逐趣成章', link: '/Hobbies/mahjong/01_fargoing' },
      { text: '更新日志', link: '/update_record' },
    ],
  },
])
