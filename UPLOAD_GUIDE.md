# GitHub 上传指南

本指南将详细说明如何将 Geo-Agent 项目上传到 GitHub 仓库。

## 步骤 1: 创建 GitHub 仓库

1. 登录 GitHub 账号: https://github.com/kiive357q
2. 点击右上角的 "+", 选择 "New repository"
3. 填写仓库信息:
   - **Repository name**: Geo-agent
   - **Description**: 智能桩基检测 Geo-Agent 系统 - 基于 PINN + RL + Transformer-CNN 的混合计算智能体
   - **Visibility**: 选择 "Private" (私密仓库)
   - 勾选 "Add a README file" (可选)
   - 点击 "Create repository"

## 步骤 2: 准备本地项目

### 检查项目结构

确保本地项目结构如下:

```
Geo-Agent/
├── checkpoints/          # 模型训练 checkpoint
├── data/                 # 数据集
├── dingo_core/           # 核心算法模块
├── geo_agent/            # 智能体核心
├── create_test_waveform.py
├── main.py
├── test_waveform.npy
├── train_dingo.py
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
└── UPLOAD_GUIDE.md
```

### 清理不必要的文件

1. 删除 `__pycache__` 目录
2. 删除 `.venv` 或 `venv` 目录（如果存在）
3. 确保 `checkpoints/` 目录下只保留必要的模型文件

## 步骤 3: 初始化 Git 仓库

在 `Geo-Agent` 目录下执行以下命令:

```bash
# 进入项目目录
cd Geo-Agent

# 初始化 Git 仓库
git init

# 配置 Git 用户名和邮箱
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加远程仓库
git remote add origin https://github.com/kiive357q/Geo-agent.git
```

## 步骤 4: 提交代码

```bash
# 添加所有文件到暂存区
git add .

# 提交代码
git commit -m "Initial commit: 智能桩基检测 Geo-Agent 系统"

# 推送代码到 GitHub
git push -u origin main
```

## 步骤 5: 验证上传结果

1. 打开 GitHub 仓库页面: https://github.com/kiive357q/Geo-agent
2. 确认所有文件已成功上传
3. 检查仓库设置，确保仓库是私密的

## 步骤 6: 配置 GitHub Actions (可选)

如果需要配置 CI/CD 流程，可以在 `.github/workflows/` 目录下创建工作流文件。

## 常见问题解决

### 1. 推送失败 - 权限问题

如果出现权限错误，请检查:
- GitHub 账号是否正确
- 是否有仓库的写入权限
- 是否需要使用 SSH 密钥认证

### 2. 推送失败 - 分支问题

如果出现分支错误，请尝试:

```bash
# 检查当前分支
git branch

# 如果分支不是 main，切换到 main
git checkout -b main

# 重新推送
git push -u origin main
```

### 3. 文件过大导致推送失败

如果某些文件过大（如模型文件），可以:
- 使用 Git LFS (Large File Storage)
- 只上传必要的模型文件
- 考虑使用 GitHub Releases 发布大文件

## 后续维护

### 更新代码

```bash
# 拉取最新代码
git pull origin main

# 修改代码后提交
git add .
git commit -m "Update: 描述修改内容"
git push origin main
```

### 创建分支

```bash
# 创建新分支
git checkout -b feature/new-feature

# 开发完成后合并到主分支
git checkout main
git merge feature/new-feature
git push origin main
```

### 标签管理

```bash
# 创建标签
git tag v1.0.0

# 推送标签
git push origin --tags
```

## 联系信息

如有问题，请联系项目维护人员。
