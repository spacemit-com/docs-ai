---
title: Runtime Management and Monitoring
sidebar_position: 10
---

# Runtime Management and Monitoring

<a id="top"></a>

Runtime is the part of BiBit that carries out tasks behind the scenes: it interprets your request, selects the appropriate model, tool, or skill, performs or blocks actions according to security policies, and records run status. The Runtime management and monitoring pages let you view and adjust these capabilities; they are intended for users who need to manage experts, model services, capabilities, or security policies. You can skip them if you only use chat or files.

## Open Runtime Pages

1. Open **Settings** and find **Management** or **Monitoring** in the sidebar.
2. Select a page. It opens on the right without leaving BiBit Settings.
3. Check the connection status at the top. Wait for **Runtime connected** before reading or changing settings.

![Runtime management and monitoring entry points in BiBit Settings](../../images/runtime-overview.png)

**If Management or Monitoring is missing:** the Runtime management extension may not be installed or enabled. Check **Settings → Extensions**, install or enable it, and restart BiBit as prompted. These pages manage advanced Runtime settings. To use **Agent Hub** or install other extensions, use their respective entry points.

## Page Overview

| Group | Page | What you can do |
| --- | --- | --- |
| Management | [Experts](experts.md) | Create, edit, and delete experts; choose their capabilities; or start a chat with an expert |
| Management | [Models](providers.md) | Add models, test the connection, switch the global model used by experts without a fixed model |
| Management | [Local Models](models.md) | View local large language and embedding model status without changing it |
| Management | [Tools](tools.md) | View agent tools and their availability |
| Management | [Skills](skills.md) | View agent skills and descriptions |
| Management | [MCP](mcp.md) | Install, update, and uninstall standalone MCP servers using launch commands |
| Management | [Plugins](plugins.md) | Install and manage Runtime backend plugins and their bundled MCP servers |
| Management | [Security](security.md) | Manage agent security modes, tool policies, file and network boundaries, system calls, DLP, and auditing |
| Monitoring | [Agent Top](monitoring.md#agent-top) | View experts, conversations, executions, and recent activity in real time |
| Monitoring | [Runtime Status](monitoring.md#runtime-status) | View the Control process and Worker pool |
| Monitoring | [Logs](monitoring.md#logs) | View and filter `runtime.jsonl` logs |

## Before You Begin

- **An accessible page does not mean every service is ready.** Check Runtime or securityd status at the top. If unavailable, refresh or check the service as prompted before submitting more changes.
- **Changes can affect agents performing tasks.** Expert configuration, tool policies, file permissions, network policies, and system call policies apply to new runs; running tasks may not change immediately.
- **A timeout does not mean nothing happened.** MCP, plugin, or directory changes can take time. If an operation times out or its result is unconfirmed, refresh the page and check the current state before deciding whether to retry.
- **Security pages are for users who need to adjust policy.** Keep defaults if you are unsure, or ask the developer community before making changes.

[Quick Start](../quickstart.md) · [FAQ](../faq.md) · [Documentation Home](../index.md)

## Related Pages

- [Experts](experts.md)
- [Models](providers.md)
- [Local Models](models.md)
- [Tools](tools.md)
- [Skills](skills.md)
- [MCP](mcp.md)
- [Plugins](plugins.md)
- [Security](security.md)
- [Runtime Monitoring](monitoring.md)

[Runtime Management and Monitoring](#top)
[BiBit User Guide](../index.md)
