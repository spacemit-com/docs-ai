sidebar_position: 11

# DeepSeek Harness

## 简介

**DeepSeek Harness** 是 DeepSeek 开源的智能体运行框架。它提供 Web 交互界面，可连接云端模型完成代码生成、文件编辑、命令执行和问题修复等任务。
本文介绍如何在 K3 的 Bianbu 系统上安装并运行 DeepSeek Harness。

## 平台支持

当前仅在 K3 Bianbu LXQT 上完成验证，其他设备与平台将在近期完成支持验证。

## 1. 安装

### 1.1 安装依赖

在 K3 终端中执行：

```
sudo apt update
sudo apt install -y npm bubblewrap build-essential python3
```

确认 Node.js 版本：
```
node -v
```

应为 22.19.0 或更高的 22.x 版本，也可以使用 Node.js 24 或更高版本，本文实测为 22.22.0。

### 1.2 安装 DeepSeek Harness

创建独立目录并安装固定版本：

```
mkdir -p "$HOME/deepseek-harness-k3"
cd "$HOME/deepseek-harness-k3"
npm install --save-exact @deepseek-ai/dsh@0.1.0-rc.6
```

第一次安装需要联网下载完整依赖树，并可能编译少量原生模块。本次 K3 冷缓存安装耗时约 22 分钟，实际时间取决于网络、存储和缓存状态。安装期间 CPU 占用较高或一段时间没有新输出不一定代表失败，请不要反复中断。

安装完成后请保留目录中的 `package-lock.json`，它记录了本次安装解析到的完整依赖版本。

### 1.3 运行

安装完成后，可直接运行以下命令启动 deepseek harness 的 Web UI

```
node --expose-internals "$HOME/deepseek-harness-k3/node_modules/.bin/dsh" web
```

看到下面的输出表示 Web 服务已启动：

```
dsh web: http://127.0.0.1:3080
```

保持当前终端运行，在设备浏览器中访问：
```
http://127.0.0.1:3080
```

即可访问 deepseek harness 的 Web UI 界面，配置完 API Key 后快乐玩耍吧 ^_^

![](../static/ds_harness_00.png)

> 若您希望将 Deepseek Harness 创建为系统命令以方便后续快速启用，可参照附录章节进行配置。

## 2. 将 K3 本地AI模型能力配置为 DSH 的插件

> 即将上线，敬请期待。

## 附录

### 创建`dsh` 命令

执行下面的命令，创建用户级 `dsh` 命令：

```
mkdir -p "$HOME/.local/bin"

printf '%s\n' \
  '#!/bin/sh' \
  'exec node --expose-internals "$HOME/deepseek-harness-k3/node_modules/@deepseek-ai/dsh/lib/bin.js" "$@"' \
  > "$HOME/.local/bin/dsh"

chmod 755 "$HOME/.local/bin/dsh"
export PATH="$HOME/.local/bin:$PATH"
```

验证配置是否正常：

```
dsh --version
```

输出以下版本号表示配置成功：

```
0.1.0-rc.6
```