import torch
print(32 * 4 * 512 )
print(192 * 1024 )

# 1. 创建输入张量 x，并开启梯度追踪
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
eps = 1e-8
z = torch.tensor([1.3, 2.2, 3.1])
# 2. 前向传播计算 y
# y = x / (sum(x) + eps)
y = x / (x.sum(dim=-1, keepdim=True) + eps)

# 3. 将向量 y 转换为标量，然后执行反向传播
# 这等价于 y.sum().backward()
y.backward(z)

# 4. 打印计算出的梯度
print("计算出的梯度 x.grad:", x.grad)

# 5. 使用我们推导的公式计算理论梯度
# 理论梯度 = eps / (sum(x) + eps)^2
S = x.sum().item() + eps
theoretical_grad = eps / (S * S)
print("理论梯度值:", theoretical_grad)
print("理论梯度向量:", torch.full_like(x, theoretical_grad))

# 6. 验证两者是否相等
print("梯度是否匹配:", torch.allclose(x.grad, torch.full_like(x, theoretical_grad)))