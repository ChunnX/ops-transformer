#!/usr/bin/env python3
"""
LeetCode 每日一题推送脚本
每天 9:15 由 cron 触发，按顺序轮换 15 个算法类别
每类选取 1-2 道经典题目，以 PDF 文件发送到"每日一题"群聊，包含：
  - 题目描述
  - 多种解法对比
  - 详细思路解析
  - 带注释的 C++ 代码
  - 触类旁通
"""

import subprocess
import sys
import os
import json
from datetime import datetime

from markdown_pdf import MarkdownPdf, Section

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, ".state")
LARK_CLI_BIN = "/home/developer/.nvm/versions/node/v18.20.8/bin/lark-cli"
GROUP_CHAT_ID = "oc_2a4c0042cd63810cf518fe7bcc430e70"


def get_pdf_path():
    """生成带日期的 PDF 文件名"""
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"每日一题_{date_str}.pdf")

# ============================================================
# 15 个类别的内容
# ============================================================
DAILY_CONTENT = [
    # ─── 第1天：贪心算法 ───
    {
        "category": "贪心算法 (Greedy)",
        "title": "分配饼干 & 无重叠区间",
        "content": """**【第1天】贪心算法 — 分配问题 & 区间问题**

## 一、算法核心思想

贪心算法的本质：**每一步都做出当前看起来最优的选择**，期望最终得到全局最优解。

贪心成立的两个条件：
1. **贪心选择性质**：局部最优能推导全局最优
2. **最优子结构**：问题的最优解包含子问题的最优解

---

## 二、经典题目：455. Assign Cookies（分配饼干）

### 题目描述
每个孩子有饥饿度 g[i]，每块饼干有饱腹度 s[j]。每个孩子最多吃一块饼干，且只有 s[j] ≥ g[i] 时孩子才能吃饱。问最多能让多少孩子吃饱。

```
输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 只有1个孩子可以吃饱（饱腹度=1的饼干喂给饥饿度=1的孩子）
```

### 解法一：排序 + 双指针（贪心）

**核心贪心策略**：优先满足饥饿度最小的孩子，并分配能让他吃饱的最小饼干。

```
为什么这样贪心是正确的？
- 如果用一块大饼干去喂饥饿度小的孩子，那饥饿度大的孩子可能就没饼干吃了
- 把最小的"足够"的饼干给饥饿度最小的孩子，保留了更大的饼干给更需要的人
```

```cpp
// 时间复杂度 O(nlogn + mlogm)，空间复杂度 O(1)
int findContentChildren(vector<int>& g, vector<int>& s) {
    // 1. 将孩子饥饿度和饼干饱腹度分别排序
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());

    int child = 0, cookie = 0;
    int n = g.size(), m = s.size();

    // 2. 双指针贪心分配
    while (child < n && cookie < m) {
        // 当前最小饼干能满足当前最小饥饿度的孩子
        if (s[cookie] >= g[child]) {
            child++;  // 孩子吃饱了，指针后移
        }
        // 饼干不管能不能喂饱，都尝试下一块
        // 如果当前饼干太小喂不饱，放弃这块饼干
        cookie++;
    }
    return child;  // 吃饱的孩子数量
}
```

---
## 三、经典题目：435. Non-overlapping Intervals（无重叠区间）

### 题目描述
给定区间集合 intervals，每个区间 [start, end]，求最少移除多少个区间，使剩余区间互不重叠。

```
输入: intervals = [[1,2],[2,3],[3,4],[1,3]]
输出: 1
解释: 移除 [1,3] 后，剩余区间互不重叠
```

### 解法：按结束时间排序 + 贪心

**核心贪心策略**：每次选择结束时间最早的区间，给后面的区间留出更多空间。

```cpp
// 时间复杂度 O(nlogn)，空间复杂度 O(1)
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;

    // 1. 按区间结束时间升序排序（贪心关键！）
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });

    int count = 0;          // 需要移除的区间数
    int prev_end = intervals[0][1];  // 上一个保留区间的结束时间

    // 2. 遍历剩余区间
    for (int i = 1; i < intervals.size(); i++) {
        // 如果当前区间与上一个保留区间重叠（起始时间 < 上一个结束时间）
        if (intervals[i][0] < prev_end) {
            count++;  // 移除当前区间
        } else {
            prev_end = intervals[i][1];  // 保留当前区间，更新结束时间
        }
    }
    return count;
}
```

### 解法对比
| 方法 | 时间复杂度 | 核心思路 |
|------|-----------|---------|
| 按结束时间排序 | O(nlogn) | 尽量选择早结束的，给后面留空间 |
| 动态规划 | O(n²) | dp[i] = 以第i个区间结尾的最长不重叠序列 |

---

## 四、触类旁通
- **452. 用最少数量的箭引爆气球** — 与"无重叠区间"异曲同工，都是按结束位置排序
- **605. 种花问题** — 遍历时贪心地种花
- **122. 买卖股票的最佳时机 II** — 只要有利润就交易（贪心累加每天正收益）""",
    },

    # ─── 第2天：双指针 ───
    {
        "category": "双指针 (Two Pointers)",
        "title": "两数之和II & 滑动窗口",
        "content": """**【第2天】双指针 — Two Sum II & 滑动窗口**

## 一、算法核心思想

双指针利用两个指针**协同移动**来遍历数据，将 O(n²) 降到 O(n)。常见模式：
- **对撞指针**：两端向中间移动（有序数组求和）
- **滑动窗口**：同向移动维护区间（子串/子数组问题）
- **快慢指针**：不同速度遍历（链表环检测）

---

## 二、经典题目：167. Two Sum II — Input Array Is Sorted

### 题目描述
在一个**升序排列**的数组中，找到两个数使它们的和等于 target。已知有且只有一对解，返回两个数的下标（1-based）。

```
输入: numbers = [2,7,11,15], target = 9
输出: [1,2]
```

### 解法：对撞指针
利用数组有序性：如果两数和太小，左指针右移增大和；两数和太大，右指针左移减小和。

```cpp
// 时间复杂度 O(n)，空间复杂度 O(1)
vector<int> twoSum(vector<int>& numbers, int target) {
    int left = 0, right = numbers.size() - 1;

    while (left < right) {
        int sum = numbers[left] + numbers[right];

        if (sum == target) {
            // 题目要求返回 1-based 下标
            return {left + 1, right + 1};
        } else if (sum < target) {
            left++;   // 和太小 → 左指针右移，增大 sum
        } else {
            right--;  // 和太大 → 右指针左移，减小 sum
        }
    }
    return {};  // 题目保证有解，此行不会执行
}
```

**正确性证明**：假设最优解是 i 和 j。当 left < i 时，sum 必然小于 target，所以 left 一直右移直到到达 i。同理 right 最终到达 j。

---

## 三、经典题目：76. Minimum Window Substring（最小覆盖子串）

### 题目描述
给定字符串 s 和 t，找出 s 中包含 t 所有字符的**最短子串**。

```
输入: s = "ADOBECODEBANC", t = "ABC"
输出: "BANC"
```

### 解法：滑动窗口

```cpp
// 时间复杂度 O(n)，空间复杂度 O(1)（字符集大小固定）
string minWindow(string s, string t) {
    // 1. 统计 t 中每个字符需要的数量
    vector<int> need(128, 0);
    for (char c : t) need[c]++;

    int left = 0, right = 0;        // 滑动窗口 [left, right)
    int count = t.size();           // 还需要匹配的字符总数
    int min_start = 0, min_len = INT_MAX;

    // 2. 右指针不断向右扩展窗口
    while (right < s.size()) {
        char c = s[right];
        right++;

        // 如果当前字符是需要的，减少需求计数
        if (need[c] > 0) count--;
        need[c]--;  // 注意：即使不需要也会减成负数

        // 3. 当窗口包含所有字符时，尝试收缩左边界
        while (count == 0) {
            // 更新最短子串
            if (right - left < min_len) {
                min_start = left;
                min_len = right - left;
            }

            // 左边界字符移出窗口
            char d = s[left];
            left++;
            need[d]++;
            // 如果移出的是必须字符，需求计数增加
            if (need[d] > 0) count++;
        }
    }
    return min_len == INT_MAX ? "" : s.substr(min_start, min_len);
}
```

### 解法对比
| 方法 | 时间复杂度 | 说明 |
|------|-----------|------|
| 暴力枚举 | O(n³) | 枚举所有子串，检查是否覆盖 |
| 滑动窗口 | O(n) | 每个字符最多被左右指针各访问一次 |

---
## 四、触类旁通
- **3. 无重复字符的最长子串** — 滑动窗口经典题
- **142. 环形链表 II** — 快慢指针（Floyd 判圈法）
- **88. 合并两个有序数组** — 逆向双指针从后往前填充""",
    },

    # ─── 第3天：二分查找 ───
    {
        "category": "二分查找 (Binary Search)",
        "title": "求平方根 & 搜索旋转数组",
        "content": """**【第3天】二分查找 — 求平方根 & 搜索旋转数组**

## 一、算法核心思想

二分查找的核心是**每次将搜索区间减半**，将时间复杂度从 O(n) 降到 O(log n)。

**使用前提**：搜索空间具有**单调性**（或可转化为单调性问题）。

**二分模板**：
- **左闭右闭 [l, r]**：`while (l <= r)`，适合精确定位
- **左闭右开 [l, r)**：`while (l < r)`，适合查找边界

---

## 二、经典题目：69. Sqrt(x)

### 题目描述
求非负整数 x 的算术平方根，向下取整。

```
输入: x = 8
输出: 2
解释: sqrt(8) = 2.828..., 向下取整得 2
```

### 解法：二分查找

在 [0, x] 范围内二分查找最大的整数 mid 满足 mid² ≤ x。

```cpp
// 时间复杂度 O(log x)，空间复杂度 O(1)
int mySqrt(int x) {
    if (x <= 1) return x;

    int left = 0, right = x;
    int ans = 0;

    // 左闭右闭二分
    while (left <= right) {
        int mid = left + (right - left) / 2;  // 防止溢出

        // 用 long long 防止 mid*mid 溢出
        if ((long long)mid * mid <= x) {
            ans = mid;        // 记录当前可行解
            left = mid + 1;   // 尝试更大的值
        } else {
            right = mid - 1;  // mid 太大了，缩小右边界
        }
    }
    return ans;
}
```

### 牛顿迭代法（扩展思路）

```cpp
// 牛顿法：x_{n+1} = (x_n + a/x_n) / 2
int mySqrtNewton(int x) {
    if (x == 0) return 0;
    double C = x, x0 = x;
    while (true) {
        double xi = 0.5 * (x0 + C / x0);
        if (fabs(x0 - xi) < 1e-7) break;
        x0 = xi;
    }
    return (int)x0;
}
```

---

## 三、经典题目：33. Search in Rotated Sorted Array

### 题目描述
一个原本升序的数组在某个点旋转后（如 [0,1,2,4,5,6,7] → [4,5,6,7,0,1,2]），在这个旋转数组中搜索 target。

```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
```

### 解法：二分查找（利用局部有序性）

**关键洞察**：虽然数组整体不是有序的，但**mid 一定将数组分成一个有序段和一个无序段**。我们只在有序段中判断 target 是否在其中。

```cpp
// 时间复杂度 O(log n)，空间复杂度 O(1)
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) return mid;

        // 判断哪一半是有序的
        if (nums[left] <= nums[mid]) {
            // 左半段有序
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;  // target 在左半段
            } else {
                left = mid + 1;   // target 在右半段
            }
        } else {
            // 右半段有序
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;   // target 在右半段
            } else {
                right = mid - 1;  // target 在左半段
            }
        }
    }
    return -1;  // 未找到
}
```

### 解法对比
| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 二分查找 | O(log n) | O(1) |
| 先找旋转点再二分 | O(log n) | O(1) |
| 线性搜索 | O(n) | O(1) |

---
## 四、触类旁通
- **34. 在排序数组中查找元素的第一个和最后一个位置** — 二分查找边界
- **162. 寻找峰值** — 二分法爬坡
- **153. 寻找旋转排序数组中的最小值** — 变种旋转数组""",
    },

    # ─── 第4天：排序算法 ───
    {
        "category": "排序算法 (Sorting)",
        "title": "快速排序 & 桶排序/第K大元素",
        "content": """**【第4天】排序算法 — 快速排序 & 第K大元素**

## 一、算法核心思想

排序是算法基础中的基础。虽然日常用 `sort()` 即可，但理解底层实现能帮你解决很多变种问题。

---

## 二、快速排序（Quick Sort）原理与实现

**核心思想**：选择一个 pivot，将数组划分为「小于 pivot」和「大于 pivot」两部分，再对两部分递归排序。

```cpp
// 快速排序 — 平均 O(nlogn)，最坏 O(n²)，空间 O(logn) 递归栈
void quickSort(vector<int>& nums, int left, int right) {
    if (left >= right) return;

    // 1. 随机选择 pivot，交换到最左边（避免最坏情况）
    int pivot_idx = left + rand() % (right - left + 1);
    swap(nums[left], nums[pivot_idx]);
    int pivot = nums[left];

    // 2. 分区操作：将小于 pivot 的放左边，大于的放右边
    int i = left + 1, j = right;
    while (true) {
        // 从左边找第一个大于等于 pivot 的元素
        while (i <= j && nums[i] < pivot) i++;
        // 从右边找第一个小于等于 pivot 的元素
        while (i <= j && nums[j] > pivot) j--;
        if (i >= j) break;
        swap(nums[i], nums[j]);
        i++; j--;
    }

    // 3. 将 pivot 放到正确位置（j 是最后一个 ≤ pivot 的位置）
    swap(nums[left], nums[j]);

    // 4. 递归排序左右两部分
    quickSort(nums, left, j - 1);
    quickSort(nums, j + 1, right);
}
```

---

## 三、经典题目：215. Kth Largest Element in an Array

### 题目描述
在未排序数组中找到第 k 个最大的元素（注意是排序后的第 k 个最大，不是第 k 个不同元素）。

```
输入: nums = [3,2,1,5,6,4], k = 2
输出: 5
```

### 解法一：快速选择（Quick Select）— 最优

快速选择是快速排序的变体：只递归包含第k大元素的那一半，平均 O(n)。

```cpp
// 时间复杂度：平均 O(n)，最坏 O(n²)；空间 O(1)
int findKthLargest(vector<int>& nums, int k) {
    // 第k大 = 第 n-k 小（0-based）
    return quickSelect(nums, 0, nums.size() - 1, nums.size() - k);
}

int quickSelect(vector<int>& nums, int left, int right, int k_smallest) {
    if (left == right) return nums[left];

    // 随机选择 pivot，避免最坏情况
    int pivot_idx = left + rand() % (right - left + 1);
    swap(nums[left], nums[pivot_idx]);
    int pivot = nums[left];

    // 分区
    int i = left + 1, j = right;
    while (true) {
        while (i <= j && nums[i] < pivot) i++;
        while (i <= j && nums[j] > pivot) j--;
        if (i >= j) break;
        swap(nums[i++], nums[j--]);
    }
    swap(nums[left], nums[j]);  // pivot 归位

    // 只递归包含目标的那一半
    if (j == k_smallest) return nums[j];
    else if (j > k_smallest) return quickSelect(nums, left, j - 1, k_smallest);
    else return quickSelect(nums, j + 1, right, k_smallest);
}
```

### 解法二：堆（优先队列）
```cpp
int findKthLargest_heap(vector<int>& nums, int k) {
    // 维护大小为 k 的小顶堆
    priority_queue<int, vector<int>, greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) minHeap.pop();  // 弹出最小的
    }
    return minHeap.top();  // 堆顶就是第k大
}
```

### 解法对比
| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 快速选择 | 平均 O(n) | O(1) | 只需一次查询 |
| 堆 | O(nlogk) | O(k) | 适合流式数据，持续查询 |
| 完全排序 | O(nlogn) | O(1) | 多次不同k查询 |

---
## 四、触类旁通
- **347. 前 K 个高频元素** — 桶排序 + 哈希表
- **75. 颜色分类** — 三路快排（荷兰国旗问题）
- **451. 根据字符出现频率排序** — 桶排序思想""",
    },

    # ─── 第5天：搜索（DFS/BFS/回溯） ───
    {
        "category": "搜索 (DFS/BFS/Backtracking)",
        "title": "岛屿最大面积 & 全排列",
        "content": """**【第5天】搜索 — 岛屿最大面积 & 全排列**

## 一、算法核心思想

- **DFS（深度优先搜索）**：一条路走到底再回溯，用**栈/递归**实现
- **BFS（广度优先搜索）**：一层一层展开，用**队列**实现
- **回溯法（Backtracking）**：DFS 的一种特殊形式，在搜索过程中**撤销选择**恢复状态

---

## 二、经典题目：695. Max Area of Island（DFS）

### 题目描述
给定二维 0-1 矩阵，0=海洋，1=陆地。相邻（上下左右）的陆地构成岛屿。求最大岛屿面积。

```
输入:
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 ...]
输出: 6
```

### 解法一：DFS 递归

```cpp
// 时间复杂度 O(mn)，空间复杂度 O(mn)（递归栈）
class Solution {
    vector<int> dir = {-1, 0, 1, 0, -1};  // 方向数组：上下左右

    int dfs(vector<vector<int>>& grid, int i, int j) {
        // 越界或非陆地，返回0
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size()
            || grid[i][j] == 0)
            return 0;

        grid[i][j] = 0;  // 标记已访问（沉岛法，省去 visited 数组）
        int area = 1;

        // 向四个方向扩展
        for (int d = 0; d < 4; d++) {
            area += dfs(grid, i + dir[d], j + dir[d+1]);
        }
        return area;
    }

public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int max_area = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    max_area = max(max_area, dfs(grid, i, j));
                }
            }
        }
        return max_area;
    }
};
```

### 解法二：BFS 队列

```cpp
int maxAreaOfIsland_BFS(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<int> dir = {-1, 0, 1, 0, -1};
    int max_area = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 0) continue;

            // BFS
            queue<pair<int,int>> q;
            q.push({i, j});
            grid[i][j] = 0;
            int area = 0;

            while (!q.empty()) {
                auto [x, y] = q.front(); q.pop();
                area++;
                for (int d = 0; d < 4; d++) {
                    int nx = x + dir[d], ny = y + dir[d+1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n
                        && grid[nx][ny] == 1) {
                        grid[nx][ny] = 0;
                        q.push({nx, ny});
                    }
                }
            }
            max_area = max(max_area, area);
        }
    }
    return max_area;
}
```

---

## 三、经典题目：46. Permutations（全排列）

### 题目描述
给定不含重复数字的数组，返回所有可能的全排列。

```
输入: nums = [1,2,3]
输出: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

### 解法：回溯法

```cpp
// 时间复杂度 O(n*n!)，空间复杂度 O(n)
class Solution {
    vector<vector<int>> result;
    vector<int> path;
    vector<bool> used;  // 标记数字是否已使用

    void backtrack(vector<int>& nums) {
        // 终止条件：排列长度等于 nums 长度
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }

        // 尝试每个未使用的数字
        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) continue;  // 剪枝：跳过已使用的

            used[i] = true;
            path.push_back(nums[i]);

            backtrack(nums);  // 递归下一层

            // 🔑 回溯：撤销选择，恢复状态
            path.pop_back();
            used[i] = false;
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        used.resize(nums.size(), false);
        backtrack(nums);
        return result;
    }
};
```

### 回溯法模板
```
void backtrack(路径, 选择列表) {
    if (满足终止条件) {
        result.add(路径);
        return;
    }
    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择   ← 这才是"回溯"的本质
}
```

---
## 四、DFS vs BFS 对比
| 特性 | DFS | BFS |
|------|-----|-----|
| 数据结构 | 栈（递归/显式栈） | 队列 |
| 最短路径 | ❌ 不保证 | ✅ 天然保证 |
| 空间 | O(h) 树高 | O(w) 层宽 |
| 适用 | 连通性、回溯、拓扑 | 最短路径、层序遍历 |

## 触类旁通
- **200. 岛屿数量** — 比最大面积更简单，DFS/BFS 模板题
- **77. 组合** — 回溯，区别是全排列讲究顺序，组合不讲究
- **79. 单词搜索** — 二维回溯 + visited 标记""",
    },

    # ─── 第6天：动态规划（一） ───
    {
        "category": "动态规划 (Dynamic Programming I)",
        "title": "爬楼梯 & 打家劫舍 & 01背包",
        "content": """**【第6天】动态规划(一) — 一维DP & 背包问题**

## 一、算法核心思想

DP 的本质：**将大问题拆成重叠子问题，存储子问题结果避免重复计算**。

DP 三要素：
1. **定义 dp 数组含义**（最关键！）
2. **推导状态转移方程**
3. **确定初始条件和遍历顺序**

---

## 二、经典题目：70. Climbing Stairs

### 题目描述
n 级台阶，每次可以走 1 步或 2 步。问多少种走法。

```
输入: n = 3
输出: 3
解释: (1,1,1), (1,2), (2,1) 三种
```

### 解法

```cpp
// 时间复杂度 O(n)，空间复杂度 O(1)
int climbStairs(int n) {
    if (n <= 2) return n;

    // dp[i] = 走到第 i 级的走法数
    // 转移方程：dp[i] = dp[i-1] + dp[i-2]
    // 因为最后一步要么从 i-1 走1步，要么从 i-2 走2步

    int prev2 = 1;  // dp[1]
    int prev1 = 2;  // dp[2]
    int curr;

    for (int i = 3; i <= n; i++) {
        curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**本质就是斐波那契数列**，用滚动变量将空间优化到 O(1)。

---

## 三、经典题目：198. House Robber（打家劫舍）

### 题目描述
一排房屋，每间有钱 nums[i]。不能偷相邻的两间。问最多能偷多少钱。

```
输入: nums = [2,7,9,3,1]
输出: 12
解释: 偷 2 + 9 + 1 = 12
```

### 解法

```cpp
// 时间复杂度 O(n)，空间复杂度 O(1)
int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    int n = nums.size();
    if (n == 1) return nums[0];

    // dp[i] = 偷前 i 间房的最大金额
    // 转移：偷第i间 = dp[i-2] + nums[i]
    //       不偷第i间 = dp[i-1]
    // dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    int prev2 = nums[0];              // dp[0]
    int prev1 = max(nums[0], nums[1]); // dp[1]

    for (int i = 2; i < n; i++) {
        int curr = max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

---

## 四、经典题目：0-1 背包问题

### 题目描述
容量为 W 的背包和 N 件物品，物品 i 重 wt[i]、价值 val[i]。每件最多选一次，求最大总价值。

```
输入: W=4, wt=[2,1,3], val=[4,2,3]
输出: 6
解释: 选第1件(2kg,4元) + 第2件(1kg,2元) = 6元
```

### 解法：二维 → 一维空间优化

```cpp
// 时间复杂度 O(N*W)，空间复杂度 O(W)
int knapsack(int W, vector<int>& wt, vector<int>& val) {
    int N = wt.size();

    // dp[w] = 容量为 w 时能获得的最大价值
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < N; i++) {
        // 🔑 逆序遍历！保证每件物品只用一次
        // 如果正序，dp[w-wt[i]] 可能已经包含第 i 件物品
        for (int w = W; w >= wt[i]; w--) {
            dp[w] = max(dp[w],                // 不选第 i 件
                        dp[w - wt[i]] + val[i]); // 选第 i 件
        }
    }
    return dp[W];
}
```

### 状态转移方程详解
```
dp[i][w] = max(dp[i-1][w],           ← 不选物品i
               dp[i-1][w-wt[i]] + val[i])  ← 选物品i

空间优化后（一维）：
dp[w] = max(dp[w], dp[w - wt[i]] + val[i])
         ↑          ↑
      上一轮的   本轮刚算的（所以必须逆序！）
```

### 背包问题变种一览
| 变种 | 特点 | 遍历顺序 |
|------|------|---------|
| 0-1 背包 | 每件最多选一次 | 逆序 |
| 完全背包 | 每件可以无限选 | 正序 |
| 多重背包 | 每件有限定次数 | 二进制拆分 |

---
## 五、触类旁通
- **213. 打家劫舍 II** — 环形房屋，拆成两个线性问题
- **322. 零钱兑换** — 完全背包问题
- **416. 分割等和子集** — 0-1背包判定问题""",
    },

    # ─── 第7天：动态规划（二） ───
    {
        "category": "动态规划 (Dynamic Programming II)",
        "title": "最长公共子序列 & 编辑距离 & 股票交易",
        "content": """**【第7天】动态规划(二) — 子序列、编辑距离 & 股票交易**

## 一、经典题目：1143. Longest Common Subsequence

### 题目描述
求两个字符串的最长公共子序列长度（子序列不要求连续，但保持相对顺序）。

```
输入: text1 = "abcde", text2 = "ace"
输出: 3
解释: "ace" 是最长公共子序列
```

### 解法：二维 DP

```cpp
// 时间复杂度 O(mn)，空间复杂度 O(mn)
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.size(), n = text2.size();

    // dp[i][j] = text1前i个字符 与 text2前j个字符 的LCS长度
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                // 字符相等，可以同时加入子序列
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                // 字符不等，取两种跳过方案的最大值
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

**空间优化**：dp[i][j] 只依赖上一行和当前行左值，可以压缩到 O(n)。

---

## 二、经典题目：72. Edit Distance（编辑距离）

### 题目描述
给定两个单词 word1 和 word2，求将 word1 转换成 word2 的最少操作数。操作：插入、删除、替换一个字符。

```
输入: word1 = "horse", word2 = "ros"
输出: 3
解释: horse -> rorse(替换h) -> rose(删r) -> ros(删e)
```

### 解法

```cpp
// 时间复杂度 O(mn)，空间 O(mn)
int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // 初始化：空串与另一个串的编辑距离 = 另一个串的长度
    for (int i = 0; i <= m; i++) dp[i][0] = i;  // 全删
    for (int j = 0; j <= n; j++) dp[0][j] = j;  // 全插

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                // 字符相同，不需要操作
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = min({
                    dp[i-1][j] + 1,    // 删除 word1[i-1]
                    dp[i][j-1] + 1,    // 插入（等效于删除word2[j-1]）
                    dp[i-1][j-1] + 1   // 替换
                });
            }
        }
    }
    return dp[m][n];
}
```

---

## 三、经典题目：121/122/123. 股票交易系列

```
121. 买卖一次 → 记录最低价格
122. 买卖无限次 → 贪心累加每天正收益
123. 最多两次 → 状态机DP
```

### 123. 最多买卖两次（通用解法）

```cpp
// 时间复杂度 O(n)，空间复杂度 O(1)
int maxProfit_2trans(vector<int>& prices) {
    // 状态定义（状态机）：
    // buy1: 完成第一次买入后的最大利润
    // sell1: 完成第一次卖出后的最大利润
    // buy2: 完成第二次买入后的最大利润
    // sell2: 完成第二次卖出后的最大利润

    int buy1 = INT_MIN, sell1 = 0;
    int buy2 = INT_MIN, sell2 = 0;

    for (int price : prices) {
        buy1  = max(buy1, -price);          // 第一次买入
        sell1 = max(sell1, buy1 + price);   // 第一次卖出
        buy2  = max(buy2, sell1 - price);   // 第二次买入
        sell2 = max(sell2, buy2 + price);   // 第二次卖出
    }
    return sell2;
}
```

### 股票问题通用框架（k 次交易）

```cpp
// 通用DP: dp[k][i] 第k次交易在第i天的最大利润
// 时间复杂度 O(kn)
int maxProfit_k(int k, vector<int>& prices) {
    int n = prices.size();
    if (n == 0) return 0;

    // k 大于 n/2 等价于无限次（122题）
    if (k > n / 2) {
        int profit = 0;
        for (int i = 1; i < n; i++)
            profit += max(0, prices[i] - prices[i-1]);
        return profit;
    }

    vector<int> buy(k + 1, INT_MIN);
    vector<int> sell(k + 1, 0);
    for (int price : prices) {
        for (int j = 1; j <= k; j++) {
            buy[j]  = max(buy[j], sell[j-1] - price);
            sell[j] = max(sell[j], buy[j] + price);
        }
    }
    return sell[k];
}
```

---
## 四、触类旁通
- **300. 最长递增子序列** — dp[i] 表示以 i 结尾的 LIS
- **516. 最长回文子序列** — 区间 DP
- **188. 买卖股票的最佳时机 IV** — 使用上面的通用框架即可""",
    },

    # ─── 第8天：分治法 ───
    {
        "category": "分治法 (Divide and Conquer)",
        "title": "为运算表达式设计优先级 & 归并排序",
        "content": """**【第8天】分治法 — 表达式求值 & 归并排序**

## 一、算法核心思想

分治 = **分**（divide） + **治**（conquer） + **合**（combine）

核心公式（主定理）：T(n) = a·T(n/b) + f(n)
- a：子问题个数
- n/b：子问题规模
- f(n)：合并开销

---

## 二、经典题目：241. Different Ways to Add Parentheses

### 题目描述
给定只含 +、-、* 的表达式，返回所有可能的加括号方式的计算结果。

```
输入: "2-1-1"
输出: [0, 2]
解释: ((2-1)-1) = 0, (2-(1-1)) = 2
```

### 解法：分治 + 组合

```cpp
// 对于每个运算符，将其左右两半分别递归求值，再组合
vector<int> diffWaysToCompute(string expression) {
    vector<int> result;

    for (int i = 0; i < expression.size(); i++) {
        char c = expression[i];
        if (c == '+' || c == '-' || c == '*') {
            // 分：递归计算左右两半的所有可能结果
            vector<int> left  = diffWaysToCompute(expression.substr(0, i));
            vector<int> right = diffWaysToCompute(expression.substr(i + 1));

            // 合：笛卡尔积组合
            for (int l : left) {
                for (int r : right) {
                    if (c == '+')      result.push_back(l + r);
                    else if (c == '-') result.push_back(l - r);
                    else if (c == '*') result.push_back(l * r);
                }
            }
        }
    }

    // 基础情况：没有运算符，直接返回数字
    if (result.empty()) {
        result.push_back(stoi(expression));
    }
    return result;
}
```

**优化**：加入 memoization（记忆化），避免重复计算相同的子表达式。

---

## 三、归并排序（Merge Sort）

经典分治：将数组分成两半 → 递归排序两半 → 合并两个有序数组。

```cpp
// 归并排序 — 稳定，O(nlogn) 时间，O(n) 空间
void mergeSort(vector<int>& nums, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    // 分：递归排序左右两半
    mergeSort(nums, left, mid);
    mergeSort(nums, mid + 1, right);

    // 治：合并两个有序子数组
    merge(nums, left, mid, right);
}

void merge(vector<int>& nums, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    // 双指针合并两个有序段
    while (i <= mid && j <= right) {
        if (nums[i] <= nums[j])
            temp[k++] = nums[i++];
        else
            temp[k++] = nums[j++];
    }

    // 把剩余元素拷入
    while (i <= mid) temp[k++] = nums[i++];
    while (j <= right) temp[k++] = nums[j++];

    // 拷回原数组
    for (int p = 0; p < temp.size(); p++)
        nums[left + p] = temp[p];
}
```

### 利用归并排序解决逆序对问题（剑指 Offer 51 / LC 493）

```cpp
// 在归并过程中统计逆序对
int reversePairs(vector<int>& nums) {
    vector<int> temp(nums.size());
    return mergeSortAndCount(nums, temp, 0, nums.size() - 1);
}

int mergeSortAndCount(vector<int>& nums, vector<int>& temp, int left, int right) {
    if (left >= right) return 0;

    int mid = left + (right - left) / 2;
    int count = mergeSortAndCount(nums, temp, left, mid)
              + mergeSortAndCount(nums, temp, mid + 1, right);

    // 合并并统计
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (nums[i] <= nums[j]) {
            temp[k++] = nums[i++];
        } else {
            temp[k++] = nums[j++];
            count += mid - i + 1;  // 🔑 左半剩余的都大于 nums[j]
        }
    }
    while (i <= mid) temp[k++] = nums[i++];
    while (j <= right) temp[k++] = nums[j++];
    for (int p = left; p <= right; p++) nums[p] = temp[p];
    return count;
}
```

### 快速排序 vs 归并排序
| 特性 | 快速排序 | 归并排序 |
|------|---------|---------|
| 平均时间 | O(nlogn) | O(nlogn) |
| 最坏时间 | O(n²) | O(nlogn) |
| 空间 | O(logn) 栈 | O(n) 临时数组 |
| 稳定性 | 不稳定 | 稳定 |
| 原地 | ✅ | ❌ 需要额外数组 |

---
## 四、触类旁通
- **215. 数组中的第K大元素** — 快速选择（分治思想的变体）
- **23. 合并K个升序链表** — 分治两两合并（归并思想）
- **95. 不同的二叉搜索树 II** — 以每个值为根，分治构造左右子树""",
    },

    # ─── 第9天：数学问题 ───
    {
        "category": "数学问题 (Math)",
        "title": "最大公约数 & 质数筛选 & 随机取样",
        "content": """**【第9天】数学问题 — 最大公约数 & 质数 & 随机取样**

## 一、经典题目：辗转相除法求 GCD/LCM

```cpp
// 辗转相除法（欧几里得算法）
// 原理：gcd(a, b) = gcd(b, a % b)
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// 最小公倍数
int lcm(int a, int b) {
    return a / gcd(a, b) * b;  // 先除后乘，防止溢出
}

// 扩展欧几里得：求 ax + by = gcd(a,b) 的解
int exgcd(int a, int b, int &x, int &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    int x1, y1;
    int d = exgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return d;
}
```

---

## 二、经典题目：204. Count Primes（计数质数）

### 题目描述
统计小于非负整数 n 的质数数量。

```
输入: n = 10
输出: 4
解释: 质数有 2, 3, 5, 7
```

### 解法：埃拉托斯特尼筛法（Sieve of Eratosthenes）

```cpp
// 时间复杂度 O(nloglogn)，空间复杂度 O(n)
int countPrimes(int n) {
    if (n <= 2) return 0;

    // isPrime[i] = true 表示 i 是质数
    vector<bool> isPrime(n, true);
    isPrime[0] = isPrime[1] = false;  // 0和1不是质数

    // 只需检查到 sqrt(n)
    for (int i = 2; i * i < n; i++) {
        if (!isPrime[i]) continue;

        // 将 i 的所有倍数标记为非质数
        // 从 i*i 开始：小于 i*i 的合数已被更小的质数标记过
        for (int j = i * i; j < n; j += i) {
            isPrime[j] = false;
        }
    }

    // 统计
    int count = 0;
    for (int i = 2; i < n; i++)
        if (isPrime[i]) count++;
    return count;
}
```

**优化要点**：
1. 内层循环从 i×i 开始（小于 i×i 的早已被标记）
2. 外层只循环到 sqrt(n)

---

## 三、经典题目：384. Shuffle an Array（洗牌算法）

### Fisher-Yates 洗牌算法

```cpp
// 每个排列等概率出现
class Solution {
    vector<int> original;
    vector<int> nums;
public:
    Solution(vector<int>& nums) : original(nums), nums(nums) {}

    vector<int> reset() {
        nums = original;
        return nums;
    }

    vector<int> shuffle() {
        int n = nums.size();

        // 从后往前，每次随机选一个元素与当前位置交换
        for (int i = n - 1; i >= 0; i--) {
            // 在 [0, i] 范围内随机选一个位置
            int j = rand() % (i + 1);
            swap(nums[i], nums[j]);
        }
        return nums;
    }
};
```

### 另一种实现：蓄水池抽样（Reservoir Sampling）

```cpp
// 从未知大小的数据流中随机选取 k 个元素
vector<int> reservoirSample(istream& stream, int k) {
    vector<int> reservoir(k);
    int val, i = 0;

    // 先填满蓄水池
    while (i < k && stream >> val) {
        reservoir[i++] = val;
    }

    // 以 k/i 的概率替换
    while (stream >> val) {
        i++;
        int j = rand() % i;  // [0, i-1]
        if (j < k) {
            reservoir[j] = val;  // 替换
        }
    }
    return reservoir;
}
```

---
## 四、触类旁通
- **7. 整数反转** — 注意溢出处理
- **168. Excel表列名称** — 26进制转换的坑（无零概念）
- **172. 阶乘后的零** — 统计因子5的个数""",
    },

    # ─── 第10天：位运算 ───
    {
        "category": "位运算 (Bit Manipulation)",
        "title": "汉明距离 & 2的幂 & 只出现一次的数字",
        "content": """**【第10天】位运算 — 汉明距离 & 只出现一次的数字**

## 一、位运算基础技巧速查表

| 操作 | 代码 | 说明 |
|------|------|------|
| 判断奇偶 | `n & 1` | 结果为1=奇数 |
| 除以2 | `n >> 1` | 右移1位 |
| 乘以2 | `n << 1` | 左移1位 |
| 最低位的1 | `n & -n` | lowbit |
| 去掉最低位的1 | `n & (n-1)` | Brian Kernighan |
| 异或性质 | `a ^ a = 0` | 相同异或得0 |
| 异或性质 | `a ^ 0 = a` | 与0异或不变 |

---

## 二、经典题目：461. Hamming Distance

### 题目描述
两个整数二进制表示中不同位的个数。

```
输入: x = 1 (0001), y = 4 (0100)
输出: 2
```

### 解法

```cpp
// 方法一：异或 + 统计1的个数
int hammingDistance(int x, int y) {
    int diff = x ^ y;  // 异或：不同位为1，相同位为0
    int dist = 0;

    // 统计 diff 中有多少个1
    while (diff) {
        dist += diff & 1;  // 判断最低位是否为1
        diff >>= 1;        // 右移一位
    }
    return dist;
}

// 方法二：利用 n & (n-1) 去除最低位的1（更快）
int hammingDistance_v2(int x, int y) {
    int diff = x ^ y, dist = 0;
    while (diff) {
        diff &= diff - 1;  // 每次去掉最低位的1
        dist++;
    }
    return dist;
}
```

---

## 三、经典题目：136. Single Number（只出现一次的数字）

### 题目描述
一个数组中，除了一个数字只出现一次外，其他每个数字都出现两次。找出那个数字。

```
输入: [4,1,2,1,2]
输出: 4
```

### 解法：异或

```cpp
// 核心：a ^ a = 0, a ^ 0 = a, 异或满足交换律和结合律
int singleNumber(vector<int>& nums) {
    int result = 0;
    // 所有数字异或起来，成对的数字相互抵消为0
    // 最终剩下的就是只出现一次的那个数
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

### 扩展：137. 只出现一次的数字 II（其他出现三次）

```cpp
// 用位运算统计每个位上1出现的次数，对3取模
int singleNumber_II(vector<int>& nums) {
    int result = 0;
    for (int i = 0; i < 32; i++) {
        int sum = 0;
        for (int num : nums) {
            sum += (num >> i) & 1;  // 统计第i位有几个1
        }
        sum %= 3;  // 出现3次的会消掉
        if (sum) result |= (1 << i);
    }
    return result;
}
```

---
## 四、经典题目：231. Power of Two（2的幂）

```cpp
// 一个数是2的幂 ⟺ 二进制中只有一个1
bool isPowerOfTwo(int n) {
    // n > 0 且 n & (n-1) == 0
    // 示例：8 = 1000，8 & 7 = 1000 & 0111 = 0
    return n > 0 && (n & (n - 1)) == 0;
}
```

### 338. Counting Bits（比特位计数）

```cpp
// dp[i] = dp[i >> 1] + (i & 1)
// i 的1的个数 = i/2 的1的个数 + i的最低位
vector<int> countBits(int n) {
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i >> 1] + (i & 1);
    }
    return dp;
}
```

---
## 五、触类旁通
- **268. 丢失的数字** — 异或或求和公式
- **260. 只出现一次的数字 III** — 分组异或
- **190. 颠倒二进制位** — 分治法翻转""",
    },

    # ─── 第11天：数据结构 — 数组/栈/队列 ───
    {
        "category": "数据结构 (Data Structures I)",
        "title": "单调栈 & 优先队列（堆）",
        "content": """**【第11天】数据结构 — 单调栈 & 优先队列**

## 一、经典题目：739. Daily Temperatures（每日温度）

### 题目描述
给定温度数组，对于每一天，求需要等多少天才能等到一个更高的温度。

```
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
```

### 解法：单调递减栈

**核心思想**：栈中存放的是尚未找到更高温度的"日期下标"，栈内温度严格递减（栈顶最小）。

```cpp
// 时间复杂度 O(n)，每个元素最多入栈出栈各一次
vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> answer(n, 0);
    stack<int> st;  // 单调递减栈，存放下标

    for (int i = 0; i < n; i++) {
        // 当前温度 > 栈顶日期温度 → 栈顶日期找到了答案
        while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
            int prev_day = st.top();
            st.pop();
            answer[prev_day] = i - prev_day;  // 等待天数
        }
        st.push(i);  // 当前日期入栈等待
    }
    // 栈中剩余的日期后面没有更高温度，answer[i] 保持为 0
    return answer;
}
```

### 单调栈通用模板
```
遇到新元素时，while 栈不为空且新元素打破单调性:
    弹出栈顶，计算栈顶的答案
    新元素入栈
```

---

## 二、经典题目：239. Sliding Window Maximum（滑动窗口最大值）

### 题目描述
给定数组 nums，滑动窗口大小 k 从左到右移动，返回每次窗口中的最大值。

```
输入: nums = [1,3,-1,-3,5,3,6,7], k = 3
输出: [3,3,5,5,6,7]
```

### 解法：单调递减双端队列

```cpp
// 时间复杂度 O(n)，空间复杂度 O(k)
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> result;
    deque<int> dq;  // 存放下标，队首始终是当前窗口的最大值

    for (int i = 0; i < nums.size(); i++) {
        // 1. 维护单调递减：队尾小于当前值的全部弹出
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }
        dq.push_back(i);

        // 2. 移除窗口外的元素（队首下标 < i-k+1）
        if (dq.front() == i - k) {
            dq.pop_front();
        }

        // 3. 窗口形成后记录结果
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    return result;
}
```

---

## 三、堆（优先队列）基本操作

```cpp
// 大顶堆（默认）
priority_queue<int> maxHeap;
maxHeap.push(3);
maxHeap.top();  // 最大值

// 小顶堆
priority_queue<int, vector<int>, greater<int>> minHeap;

// 自定义比较（比如按pair的第二个元素）
auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
    return a.second > b.second;  // 小顶堆
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
```

### 经典题：347. Top K Frequent Elements

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    // 1. 统计频率
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;

    // 2. 小顶堆（按频率排序）
    auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> minHeap(cmp);

    // 3. 维护大小为 k 的堆
    for (auto& [num, count] : freq) {
        minHeap.push({num, count});
        if (minHeap.size() > k) minHeap.pop();
    }

    // 4. 输出结果
    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().first);
        minHeap.pop();
    }
    return result;
}
```

### 单调栈 vs 优先队列 vs 双端队列
| 数据结构 | 用途 | 时间复杂度 |
|---------|------|-----------|
| 单调栈 | 找下一个更大/更小元素 | O(n) |
| 优先队列(堆) | 动态维护最大/最小值 | O(log n)/操作 |
| 双端队列 | 滑动窗口最大值 | O(n) |

---
## 四、触类旁通
- **84. 柱状图中最大的矩形** — 单调栈经典难题
- **295. 数据流的中位数** — 双堆（大顶堆+小顶堆）
- **42. 接雨水** — 单调栈/双指针两种解法""",
    },

    # ─── 第12天：哈希表 & 前缀和 ───
    {
        "category": "数据结构 (Data Structures II)",
        "title": "哈希表 & 前缀和 & 积分图",
        "content": """**【第12天】数据结构(二) — 哈希表 & 前缀和**

## 一、经典题目：1. Two Sum

### 题目描述
在数组中找出两个数，使它们的和等于 target。返回下标。

```
输入: nums = [2,7,11,15], target = 9
输出: [0,1]
```

### 解法：哈希表一次遍历

```cpp
// 时间复杂度 O(n)，空间复杂度 O(n)
vector<int> twoSum(vector<int>& nums, int target) {
    // key=数值, value=下标
    unordered_map<int, int> map;

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];

        // 检查之前是否见过 target - nums[i]
        if (map.find(complement) != map.end()) {
            return {map[complement], i};
        }
        map[nums[i]] = i;
    }
    return {};
}
```

---

## 二、经典题目：560. Subarray Sum Equals K

### 题目描述
和为 K 的连续子数组的个数。

```
输入: nums = [1,2,3], k = 3
输出: 2
解释: [1,2] 和 [3]
```

### 解法：前缀和 + 哈希表

**核心思路**：子数组 nums[i..j] 的和 = prefix[j] - prefix[i-1]

```cpp
// 求 prefix[j] - prefix[i-1] = k 即 prefix[i-1] = prefix[j] - k
int subarraySum(vector<int>& nums, int k) {
    // 哈希表记录每种前缀和出现的次数
    unordered_map<int, int> prefix_count;
    prefix_count[0] = 1;  // 空前缀和为0，出现1次（用于处理子数组从头开始的情况）

    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;  // 当前前缀和

        // 如果之前有前缀和为 sum - k，那么中间的子数组和就是 k
        if (prefix_count.find(sum - k) != prefix_count.end()) {
            count += prefix_count[sum - k];
        }
        prefix_count[sum]++;
    }
    return count;
}
```

**图解**：
```
nums = [1, 2, 3], k = 3
i=0: sum=1, 找 prefix-sum=1-3=-2, 没有 → map={0:1, 1:1}
i=1: sum=3, 找 prefix-sum=3-3=0, 有! count+=1 → map={0:1, 1:1, 3:1}
i=2: sum=6, 找 prefix-sum=6-3=3, 有! count+=1 → map={0:1, 1:1, 3:1, 6:1}
count = 2
```

---

## 三、二维前缀和（积分图）

```cpp
// 304. Range Sum Query 2D - Immutable
class NumMatrix {
    // sum[i][j] = 从(0,0)到(i-1,j-1)的矩阵和
    vector<vector<int>> sum;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        sum = vector<vector<int>>(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 前缀和公式
                sum[i][j] = matrix[i-1][j-1]
                          + sum[i-1][j]        // 上方
                          + sum[i][j-1]        // 左方
                          - sum[i-1][j-1];     // 减去重复的左上角
            }
        }
    }

    // 查询 (r1,c1) 到 (r2,c2) 的矩阵和
    int sumRegion(int r1, int c1, int r2, int c2) {
        return sum[r2+1][c2+1]
             - sum[r1][c2+1]    // 减去上方
             - sum[r2+1][c1]    // 减去左方
             + sum[r1][c1];     // 加回重复减去的左上角
    }
};
```

---
## 四、触类旁通
- **15. 三数之和** — 排序+双指针（去重是关键）
- **128. 最长连续序列** — 哈希集合 O(n)
- **525. 连续数组** — 前缀和 + 哈希表（0和1数量相等）""",
    },

    # ─── 第13天：字符串 ───
    {
        "category": "字符串 (String)",
        "title": "字符串比较 & KMP算法",
        "content": """**【第13天】字符串 — 字符串比较 & KMP 匹配**

## 一、经典题目：242. Valid Anagram（有效字母异位词）

```cpp
// 时间复杂度 O(n)，空间 O(1)
bool isAnagram(string s, string t) {
    if (s.size() != t.size()) return false;

    vector<int> count(26, 0);
    for (int i = 0; i < s.size(); i++) {
        count[s[i] - 'a']++;   // s中字符贡献正数
        count[t[i] - 'a']--;   // t中字符贡献负数
    }

    // 如果所有计数归零，说明两个字符串字符组成完全相同
    for (int c : count)
        if (c != 0) return false;
    return true;
}
```

---

## 二、经典题目：409. Longest Palindrome（最长回文串）

```
输入: "abccccdd"
输出: 7
解释: 可以构造 "dccaccd"（7个字符的回文串）
```

```cpp
int longestPalindrome(string s) {
    unordered_map<char, int> count;
    for (char c : s) count[c]++;

    int length = 0;
    bool hasOdd = false;

    for (auto& [ch, cnt] : count) {
        length += cnt / 2 * 2;  // 偶数部分全部可以用
        if (cnt % 2 == 1) hasOdd = true;
    }

    // 如果存在奇数个字符，可以放一个在正中间
    return hasOdd ? length + 1 : length;
}
```

---

## 三、KMP 字符串匹配算法

### 28. Implement strStr()

```cpp
// KMP算法 — 时间复杂度 O(m+n)，空间复杂度 O(n)
// 其中 m = haystack长度, n = needle长度

// 构建 next 数组（前缀表）
// next[i] = needle[0..i] 的最长相等前后缀的长度
vector<int> buildNext(string& pattern) {
    int n = pattern.size();
    vector<int> next(n, 0);
    int j = 0;  // j = 前缀末尾位置，也等于当前最长相等前后缀长度

    for (int i = 1; i < n; i++) {
        // 不匹配时回退
        while (j > 0 && pattern[i] != pattern[j]) {
            j = next[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            j++;
        }
        next[i] = j;
    }
    return next;
}

int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;
    int m = haystack.size(), n = needle.size();

    // 1. 构建 next 数组
    vector<int> next = buildNext(needle);

    // 2. 进行匹配
    int j = 0;  // needle 的指针
    for (int i = 0; i < m; i++) {
        // 不匹配时，利用 next 数组回退（核心！）
        while (j > 0 && haystack[i] != needle[j]) {
            j = next[j - 1];
        }
        if (haystack[i] == needle[j]) {
            j++;
        }
        // 完全匹配
        if (j == n) {
            return i - n + 1;
        }
    }
    return -1;
}
```

**KMP 核心思想**：当发生不匹配时，利用已经匹配部分的"最长相等前后缀"信息，将模式串跳过不可能匹配的位置，避免主串指针回退。

### 示例解析
```
haystack = "aabaabaafa"
needle   = "aabaaf"

next数组: [0,1,0,1,2,0]
           a a b a a f

匹配过程:
a a b a a b a a f a      ← 到 b != f 时，
a a b a a f              ← j回退到 next[4]=2
      ↓
a a b a a b a a f a
      a a b a a f        ← 从 aab 继续匹配
```

---
## 四、触类旁通
- **3. 无重复字符的最长子串** — 滑动窗口 + HashSet
- **5. 最长回文子串** — 中心扩展法 O(n²) 或 Manacher O(n)
- **49. 字母异位词分组** — 排序字符串作 key""",
    },

    # ─── 第14天：链表 & 树 ───
    {
        "category": "链表 & 树 (Linked List & Tree)",
        "title": "链表翻转 & 二叉树遍历 & 二叉搜索树",
        "content": """**【第14天】链表 & 树 — 翻转链表 & 二叉树遍历 & BST**

## 一、206. Reverse Linked List（翻转链表）

### 迭代法

```cpp
// 时间复杂度 O(n)，空间 O(1)
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;

    while (curr) {
        ListNode* next = curr->next;  // 暂存下一个节点
        curr->next = prev;            // 反转指针
        prev = curr;                  // prev 前进
        curr = next;                  // curr 前进
    }
    return prev;  // prev 就是新的头节点
}
// 图解:   null ← 1 ← 2 ← 3
//         prev  curr next
```

### 递归法

```cpp
ListNode* reverseList_recursive(ListNode* head) {
    // 基础条件：空链表或单节点
    if (!head || !head->next) return head;

    // 递归翻转后面的部分，newHead 是最终的头节点
    ListNode* newHead = reverseList_recursive(head->next);

    // 将当前节点接到后面节点的后面
    head->next->next = head;  // 让下一个节点指向自己
    head->next = nullptr;     // 自己的 next 置空

    return newHead;
}
```

---

## 二、二叉树遍历（三种统一定义）

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
};

// 二叉树最大深度 (104)
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

// 判断平衡二叉树 (110)
bool isBalanced(TreeNode* root) {
    if (!root) return true;
    return abs(maxDepth(root->left) - maxDepth(root->right)) <= 1
        && isBalanced(root->left)
        && isBalanced(root->right);
}

// 二叉树的直径 (543) — 后序遍历
// 直径 = 某个节点左右子树深度之和的最大值
int diameterOfBinaryTree(TreeNode* root) {
    int max_diameter = 0;
    function<int(TreeNode*)> depth = [&](TreeNode* node) {
        if (!node) return 0;
        int left = depth(node->left);
        int right = depth(node->right);
        max_diameter = max(max_diameter, left + right);  // 更新直径
        return 1 + max(left, right);                     // 返回深度
    };
    depth(root);
    return max_diameter;
}
```

---

## 三、98. Validate Binary Search Tree

### 题目描述
判断一棵二叉树是否是有效的二叉搜索树（左 < 根 < 右）。

### 解法一：中序遍历（利用 BST 中序遍历严格递增）

```cpp
// 时间复杂度 O(n)，空间 O(n)
bool isValidBST(TreeNode* root) {
    long long prev = LLONG_MIN;
    return inorder(root, prev);
}

bool inorder(TreeNode* node, long long& prev) {
    if (!node) return true;

    // 左
    if (!inorder(node->left, prev)) return false;

    // 根：检查是否严格递增
    if (node->val <= prev) return false;
    prev = node->val;

    // 右
    return inorder(node->right, prev);
}
```

### 解法二：带上下界的 DFS

```cpp
bool isValidBST_v2(TreeNode* root) {
    return dfs(root, nullptr, nullptr);
}

bool dfs(TreeNode* node, TreeNode* minNode, TreeNode* maxNode) {
    if (!node) return true;

    // 检查当前节点值是否在合法范围内
    if (minNode && node->val <= minNode->val) return false;
    if (maxNode && node->val >= maxNode->val) return false;

    // 左子树：上界变为当前节点
    // 右子树：下界变为当前节点
    return dfs(node->left, minNode, node)
        && dfs(node->right, node, maxNode);
}
```

### 树遍历方法对比
| 遍历方式 | 顺序 | 常用场景 |
|---------|------|---------|
| 前序 | 根→左→右 | 序列化、复制 |
| 中序 | 左→根→右 | BST输出有序序列 |
| 后序 | 左→右→根 | 自底向上计算（如直径、高度） |
| 层序 | 逐层 | BFS、最短路径 |

---
## 四、触类旁通
- **141. 环形链表** — 快慢指针
- **21. 合并两个有序链表** — 归并思想
- **236. 二叉树的最近公共祖先** — 后序遍历""",
    },

    # ─── 第15天：图 & 并查集 ───
    {
        "category": "图 & 并查集 (Graph & Union-Find)",
        "title": "二分图 & 拓扑排序 & 并查集",
        "content": """**【第15天】图 & 并查集 — 二分图 & 拓扑排序 & 并查集**

## 一、经典题目：785. Is Graph Bipartite?（二分图判定）

### 题目描述
判断无向图是否是二分图（可以分成两组，每组内部无边）。

```
输入: graph = [[1,3],[0,2],[1,3],[0,2]]
解释: 0---1
     |   |
     3---2
可以分成 {0,2} 和 {1,3} → true
```

### 解法：染色法 BFS

```cpp
// 时间复杂度 O(V+E)，空间复杂度 O(V)
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    // colors: 0=未染色, 1=颜色A, 2=颜色B
    vector<int> colors(n, 0);

    // 图可能不连通，需要遍历所有节点
    for (int i = 0; i < n; i++) {
        if (colors[i] != 0) continue;  // 已经染过色的跳过

        // BFS 染色
        queue<int> q;
        q.push(i);
        colors[i] = 1;  // 起点染颜色1

        while (!q.empty()) {
            int node = q.front(); q.pop();
            int expectColor = (colors[node] == 1) ? 2 : 1;

            for (int neighbor : graph[node]) {
                if (colors[neighbor] == 0) {
                    colors[neighbor] = expectColor;
                    q.push(neighbor);
                } else if (colors[neighbor] != expectColor) {
                    // 邻居颜色冲突 → 不是二分图
                    return false;
                }
            }
        }
    }
    return true;
}
```

---

## 二、经典题目：207. Course Schedule（课程表/拓扑排序）

### 题目描述
n 门课，prerequisites[i] = [a, b] 表示学 a 前必须先学 b。问是否可以完成所有课程（无环）。

```
输入: n = 2, prerequisites = [[1,0]]
输出: true
```

### 解法一：BFS 拓扑排序（Kahn 算法）

```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    // 构建邻接表和入度数组
    vector<vector<int>> graph(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& pre : prerequisites) {
        graph[pre[1]].push_back(pre[0]);  // b → a
        indegree[pre[0]]++;               // a 的入度+1
    }

    // 将所有入度为0的节点入队
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.push(i);
    }

    int visited = 0;  // 已遍历节点数
    while (!q.empty()) {
        int node = q.front(); q.pop();
        visited++;

        // 移除当前节点，将其后继节点的入度-1
        for (int next : graph[node]) {
            indegree[next]--;
            if (indegree[next] == 0) {
                q.push(next);
            }
        }
    }

    // 如果所有节点都被访问 → 无环
    return visited == numCourses;
}
```

### 解法二：DFS 三色标记法
```
白色(0)=未访问, 灰色(1)=正在访问, 黑色(2)=已完成
如果在DFS过程中遇到灰色节点 → 存在环
```

```cpp
bool canFinish_DFS(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    for (auto& pre : prerequisites)
        graph[pre[1]].push_back(pre[0]);

    vector<int> state(numCourses, 0);  // 0=白, 1=灰, 2=黑

    function<bool(int)> dfs = [&](int node) {
        state[node] = 1;  // 标记为正在访问（灰色）
        for (int next : graph[node]) {
            if (state[next] == 1) return false;  // 遇到灰色 → 环
            if (state[next] == 0 && !dfs(next)) return false;
        }
        state[node] = 2;  // 标记为已完成（黑色）
        return true;
    };

    for (int i = 0; i < numCourses; i++) {
        if (state[i] == 0 && !dfs(i)) return false;
    }
    return true;
}
```

---

## 三、并查集（Union-Find）

```cpp
class UnionFind {
    vector<int> parent;
    vector<int> rank;  // 按秩合并优化
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    // 查找根节点 + 路径压缩
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // 路径压缩
        return parent[x];
    }

    // 合并两个集合
    void unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;

        // 按秩合并：把小树接在大树下
        if (rank[rx] < rank[ry])
            parent[rx] = ry;
        else if (rank[rx] > rank[ry])
            parent[ry] = rx;
        else {
            parent[ry] = rx;
            rank[rx]++;
        }
    }

    // 判断是否连通
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

### 并查集复杂度
| 操作 | 时间复杂度 |
|------|-----------|
| find（路径压缩） | 近似 O(1)（阿克曼函数反函数） |
| unite（按秩合并） | 近似 O(1) |

---
## 四、触类旁通
- **210. 课程表 II** — 输出拓扑排序结果
- **684. 冗余连接** — 并查集找环
- **547. 省份数量** — 并查集统计连通分量""",
    },
]


def get_day_index():
    """读取当前应该发送第几天的内容"""
    try:
        with open(STATE_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def save_day_index(day):
    """保存当前天数"""
    with open(STATE_FILE, "w") as f:
        f.write(str(day))


def build_message(day_index):
    """构建第 day_index 天的消息内容"""
    if day_index >= len(DAILY_CONTENT):
        day_index = 0  # 循环

    data = DAILY_CONTENT[day_index]
    category = data["category"]
    title = data["title"]
    body = data["content"]

    date_str = datetime.now().strftime("%Y年%m月%d日")

    header = f"## LeetCode 每日一题 ({date_str})\n\n### 今日主题：{category} — {title}\n\n"

    return header + body


def build_text_preview(day_index):
    """构建文本预览消息"""
    data = DAILY_CONTENT[day_index % len(DAILY_CONTENT)]
    date_str = datetime.now().strftime("%Y年%m月%d日")
    return f"每日一题 ({date_str})\n今日主题：{data['category']} — {data['title']}\n详细题解请查看附件 PDF"


CODE_CSS = """
pre {
    background-color: #f0f0f0;
    border: 1px solid #c0c0c0;
    border-radius: 3px;
    padding: 8px 10px;
    margin: 8px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    font-size: 8pt;
    line-height: 1.4;
}
pre code {
    font-family: monospace;
    font-size: 8pt;
    line-height: 1.4;
}
"""


def markdown_to_pdf(markdown_text, pdf_path):
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(markdown_text.strip()), user_css=CODE_CSS)
    pdf.save(pdf_path)
    return pdf_path


def send_message(markdown_text, text_preview):
    """先生成 PDF，再通过 lark-cli 发送到群聊"""
    pdf_path = get_pdf_path()

    # 1. 生成 PDF
    markdown_to_pdf(markdown_text, pdf_path)

    # 2. 先发送文本预览
    cmd_text = [
        LARK_CLI_BIN, "im", "+messages-send",
        "--chat-id", GROUP_CHAT_ID,
        "--text", text_preview,
        "--as", "bot",
    ]
    result_text = subprocess.run(cmd_text, capture_output=True, text=True, timeout=30)

    # 3. 发送 PDF 文件（上传大文件需要更长时间）
    pdf_filename = os.path.basename(pdf_path)
    cmd_pdf = [
        LARK_CLI_BIN, "im", "+messages-send",
        "--chat-id", GROUP_CHAT_ID,
        "--file", pdf_filename,
        "--as", "bot",
    ]
    result_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True, timeout=180, cwd=SCRIPT_DIR)

    if result_pdf.returncode != 0:
        print(f"发送 PDF 失败: {result_pdf.stderr}", file=sys.stderr)
        return False

    try:
        resp = json.loads(result_pdf.stdout)
        if resp.get("ok"):
            msg_id = resp["data"]["message_id"]
            print(f"PDF 发送成功! message_id={msg_id}", file=sys.stderr)
            return True
        else:
            print(f"发送失败: {result_pdf.stdout}", file=sys.stderr)
            return False
    except json.JSONDecodeError:
        print(f"解析失败: {result_pdf.stdout}", file=sys.stderr)
        return False


def main():
    day = get_day_index()
    day_data = DAILY_CONTENT[day % len(DAILY_CONTENT)]
    print(f"当前天数: {day}, 发送类别: {day_data['category']}")

    message = build_message(day)
    preview = build_text_preview(day)
    success = send_message(message, preview)

    if success:
        save_day_index((day + 1) % len(DAILY_CONTENT))
        print(f"已更新天数到 {(day + 1) % len(DAILY_CONTENT)}")


if __name__ == "__main__":
    main()
