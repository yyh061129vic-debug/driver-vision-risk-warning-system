# 组员提交区

本目录用于组员分别提交阶段成果、实验记录和代码说明，避免多人直接修改同一文件造成冲突。

## 目录约定

每位组员在本目录下创建一个以 GitHub 用户名命名的子目录：

```text
team_submissions/
├─ README.md
├─ example/
│  └─ README.md
└─ <github-username>/
   ├─ README.md
   ├─ configs/
   ├─ notes/
   └─ scripts/
```

组员只应把个人实验配置、说明和新增脚本放入自己的目录。需要进入项目正式模块的代码，应通过 Pull Request 提出，并由负责人审查后合并到 `src/`、`configs/` 或 `scripts/`。

## 提交流程

1. 从最新 `main` 创建个人分支，例如 `member/<github-username>-task-name`。
2. 在 `team_submissions/<github-username>/` 下保存本次成果。
3. 运行与改动相关的校验或测试。
4. 提交并推送个人分支。
5. 向 `main` 创建 Pull Request，写明改动、目的、验证结果和已知限制。

## 禁止提交

- `data_raw/` 和 `data_processed/` 中的数据本体；
- `checkpoints/` 中的模型权重；
- `outputs/` 中的运行产物；
- `.venv/`、密钥、令牌、个人绝对路径和机器敏感信息；
- 许可证不允许再分发的图像、视频、标注或衍生可视化。

组员没有仓库写权限时，应 Fork 仓库并从个人 Fork 创建 Pull Request；只有仓库管理员可以在 GitHub 设置中邀请 Collaborator。
