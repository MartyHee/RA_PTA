#!/usr/bin/env python3
"""
抖音开放平台API演示脚本。

功能：
1. OAuth授权演示
2. 用户信息获取演示
3. 视频数据查询演示
4. 数据合并演示
5. Mock模式（无需真实API凭证）

注意：使用真实API需要：
1. 注册抖音开放平台开发者
2. 创建应用获取client_key和client_secret
3. 申请相应权限（user_info, video.data等）
"""
import sys
import argparse
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.auth import DouyinAuth as DouyinOAuth
from src.api.client import DouyinAPIClient
from src.api.video_data import DouyinVideoData as VideoDataAPI
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging, get_logger
from src.schemas.tables import ApiVideoStats, ApiUserProfile


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抖音开放平台API演示')
    parser.add_argument('--mode', choices=['auth', 'user', 'video', 'merge', 'mock'],
                       default='mock', help='运行模式')
    parser.add_argument('--code', type=str, help='OAuth授权码（用于auth模式）')
    parser.add_argument('--open-id', type=str, help='用户OpenID')
    parser.add_argument('--video-id', type=str, help='视频ID')
    parser.add_argument('--web-data', type=Path, help='网页数据文件路径（用于合并）')
    parser.add_argument('--output', type=Path, help='输出文件路径')
    parser.add_argument('--config', type=Path, default=None, help='配置文件路径')
    parser.add_argument('--list-videos', action='store_true', help='列出用户视频（需要video.list权限）')
    parser.add_argument('--count', type=int, default=10, help='视频数量（用于列表）')
    return parser.parse_args()


def setup_api_client(config: Dict[str, Any], mode: str) -> Optional[DouyinAPIClient]:
    """设置API客户端"""
    if mode == 'mock':
        print("使用Mock模式，无需真实API凭证")
        return None

    api_config = config.get('api', {}).get('douyin', {})
    if not api_config.get('enabled', False):
        print("API功能未启用，请在configs/settings.yaml中启用")
        return None

    # 检查环境变量或配置文件中的凭证
    client_key = api_config.get('client_key')
    client_secret = api_config.get('client_secret')

    if not client_key or not client_secret:
        print("未找到API凭证，请设置DOUYIN_CLIENT_KEY和DOUYIN_CLIENT_SECRET环境变量")
        return None

    return DouyinAPIClient(client_key=client_key, client_secret=client_secret)


def demo_oauth_flow(config: Dict[str, Any], code: Optional[str] = None):
    """演示OAuth授权流程"""
    print("=" * 60)
    print("OAuth授权流程演示")
    print("=" * 60)

    oauth = DouyinOAuth(config_path=args.config)

    if code:
        # 步骤3：使用授权码换取访问令牌
        print(f"使用授权码换取访问令牌...")
        try:
            token_data = oauth.get_access_token(code)
            print("授权成功!")
            print(f"OpenID: {token_data.get('open_id')}")
            print(f"Access Token: {token_data.get('access_token')[:20]}...")
            print(f"Expires in: {token_data.get('expires_in')}秒")
            print(f"Refresh Token: {token_data.get('refresh_token')[:20]}...")
            print(f"Scope: {token_data.get('scope')}")

            # 保存令牌（示例）
            import json
            token_file = Path('data/temp/douyin_token.json')
            token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            print(f"令牌已保存到: {token_file}")

            return token_data
        except Exception as e:
            print(f"获取访问令牌失败: {e}")
            return None
    else:
        # 步骤1：生成授权URL
        auth_url = oauth.get_authorization_url()
        print(f"请访问以下URL进行授权:")
        print(auth_url)
        print()
        print("授权后，您将被重定向到回调地址")
        print("请从回调URL中获取code参数，并使用 --code 参数运行此脚本")

        # 尝试在浏览器中打开
        try:
            webbrowser.open(auth_url)
            print("已在浏览器中打开授权页面")
        except:
            print("无法自动打开浏览器，请手动复制URL访问")

        return None


def demo_user_info(config: Dict[str, Any], open_id: str):
    """演示用户信息获取"""
    print("=" * 60)
    print("用户信息获取演示")
    print("=" * 60)

    client = setup_api_client(config, 'mock' if args.mode == 'mock' else 'api')

    if client is None and args.mode != 'mock':
        print("无法创建API客户端")
        return

    if args.mode == 'mock':
        # Mock响应
        print("使用Mock数据演示...")
        mock_user = {
            "open_id": open_id or "mock_open_id_123",
            "nickname": "抖音测试用户",
            "avatar_url": "https://example.com/avatar.jpg",
            "gender": "male",
            "province": "北京市",
            "city": "北京市",
            "pull_time": pd.Timestamp.now()
        }

        # 验证数据模式
        try:
            user_profile = ApiUserProfile(**mock_user)
            print("用户信息（验证通过）:")
            for key, value in user_profile.dict().items():
                print(f"  {key}: {value}")

            # 保存示例
            if args.output:
                df = pd.DataFrame([user_profile.dict()])
                df.to_parquet(args.output, index=False)
                print(f"用户信息已保存到: {args.output}")
        except Exception as e:
            print(f"数据验证失败: {e}")
    else:
        # 真实API调用
        print("调用真实API...")
        try:
            user_info = client.get_user_info(open_id)
            print("用户信息:")
            for key, value in user_info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"获取用户信息失败: {e}")


def demo_video_data(config: Dict[str, Any], video_id: str):
    """演示视频数据获取"""
    print("=" * 60)
    print("视频数据获取演示")
    print("=" * 60)

    client = setup_api_client(config, 'mock' if args.mode == 'mock' else 'api')

    if client is None and args.mode != 'mock':
        print("无法创建API客户端")
        return

    if args.mode == 'mock':
        # Mock响应
        print("使用Mock数据演示...")
        mock_video = {
            "video_id": video_id or "mock_video_123",
            "open_id": "mock_open_id_123",
            "stat_time": pd.Timestamp.now(),
            "play_count": 15000,
            "digg_count": 1200,
            "comment_count": 85,
            "share_count": 45,
            "cover_url": "https://example.com/cover.jpg",
            "create_time": pd.Timestamp.now() - pd.Timedelta(days=7),
            "api_pull_time": pd.Timestamp.now()
        }

        # 验证数据模式
        try:
            video_stats = ApiVideoStats(**mock_video)
            print("视频统计（验证通过）:")
            for key, value in video_stats.dict().items():
                print(f"  {key}: {value}")

            # 计算互动分数（基于API数据）
            engagement = video_stats.digg_count * 1.0 + video_stats.comment_count * 2.0 + video_stats.share_count * 3.0
            print(f"  互动分数: {engagement:.1f}")

            # 保存示例
            if args.output:
                df = pd.DataFrame([video_stats.dict()])
                df.to_parquet(args.output, index=False)
                print(f"视频数据已保存到: {args.output}")
        except Exception as e:
            print(f"数据验证失败: {e}")
    else:
        # 真实API调用
        print("调用真实API...")
        try:
            video_data = client.get_video_data(video_id)
            print("视频数据:")
            for key, value in video_data.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"获取视频数据失败: {e}")


def demo_list_videos(config: Dict[str, Any], open_id: str):
    """演示用户视频列表获取"""
    print("=" * 60)
    print("用户视频列表获取演示")
    print("=" * 60)

    client = setup_api_client(config, 'mock' if args.mode == 'mock' else 'api')

    if client is None and args.mode != 'mock':
        print("无法创建API客户端")
        return

    if args.mode == 'mock':
        # Mock响应
        print("使用Mock数据演示...")
        mock_videos = []
        for i in range(min(args.count, 10)):
            mock_videos.append({
                "item_id": f"mock_video_{i+1}",
                "desc": f"测试视频描述 {i+1}",
                "cover_url": f"https://example.com/cover_{i+1}.jpg",
                "create_time": pd.Timestamp.now() - pd.Timedelta(days=i),
                "digg_count": 100 * (i + 1),
                "comment_count": 10 * (i + 1),
                "share_count": 5 * (i + 1)
            })

        print(f"获取到 {len(mock_videos)} 个视频:")
        for i, video in enumerate(mock_videos[:5]):  # 只显示前5个
            print(f"  {i+1}. {video['item_id']}: {video['desc'][:30]}...")

        if len(mock_videos) > 5:
            print(f"  ... 还有 {len(mock_videos) - 5} 个视频")

        # 保存示例
        if args.output:
            df = pd.DataFrame(mock_videos)
            df.to_parquet(args.output, index=False)
            print(f"视频列表已保存到: {args.output}")
    else:
        # 真实API调用
        print("调用真实API...")
        try:
            videos = client.get_user_videos(open_id, count=args.count)
            print(f"获取到 {len(videos)} 个视频:")
            for i, video in enumerate(videos[:5]):  # 只显示前5个
                print(f"  {i+1}. {video.get('item_id')}: {video.get('desc', '')[:30]}...")

            if len(videos) > 5:
                print(f"  ... 还有 {len(videos) - 5} 个视频")
        except Exception as e:
            print(f"获取视频列表失败: {e}")
            print("可能需要 video.list 权限")


def demo_data_merge(config: Dict[str, Any], web_data_path: Path):
    """演示网页数据与API数据合并"""
    print("=" * 60)
    print("数据合并演示")
    print("=" * 60)

    # 加载网页数据
    if not web_data_path.exists():
        print(f"网页数据文件不存在: {web_data_path}")
        print("请先运行爬虫获取网页数据")
        return

    try:
        if web_data_path.suffix == '.parquet':
            web_df = pd.read_parquet(web_data_path)
        elif web_data_path.suffix == '.csv':
            web_df = pd.read_csv(web_data_path)
        else:
            print(f"不支持的文件格式: {web_data_path.suffix}")
            return
    except Exception as e:
        print(f"加载网页数据失败: {e}")
        return

    print(f"加载网页数据: {len(web_df)} 条记录")
    print(f"字段: {', '.join(web_df.columns[:5])}...")

    if args.mode == 'mock':
        # Mock API数据
        print("生成Mock API数据...")
        api_data = []
        for _, row in web_df.head(args.count).iterrows():  # 只处理前N条
            video_id = row.get('video_id', f"mock_{_}")
            api_data.append({
                'video_id': video_id,
                'open_id': 'mock_open_id_123',
                'stat_time': pd.Timestamp.now(),
                'play_count': int(row.get('like_count', 0) * 50),  # 假设播放数是点赞数的50倍
                'digg_count': row.get('like_count'),
                'comment_count': row.get('comment_count'),
                'share_count': row.get('share_count'),
                'cover_url': row.get('cover_url'),
                'create_time': row.get('publish_time_std'),
                'api_pull_time': pd.Timestamp.now(),
                'data_source': 'mock_api'
            })

        api_df = pd.DataFrame(api_data)
        print(f"生成API数据: {len(api_df)} 条记录")

        # 合并数据
        merged_df = pd.merge(
            web_df.head(args.count),
            api_df[['video_id', 'play_count', 'api_pull_time', 'data_source']],
            on='video_id',
            how='left',
            suffixes=('_web', '_api')
        )

        # 添加合并标记
        merged_df['data_source'] = merged_df['data_source'].fillna('web_only')
        merged_df['play_count'] = merged_df['play_count'].fillna(-1)  # -1表示无API数据

        print("合并结果:")
        print(f"  总记录数: {len(merged_df)}")
        print(f"  有API数据: {(merged_df['data_source'] == 'mock_api').sum()}")
        print(f"  仅网页数据: {(merged_df['data_source'] == 'web_only').sum()}")

        # 显示示例
        print("\n合并数据示例:")
        sample_cols = ['video_id', 'like_count', 'play_count', 'data_source']
        print(merged_df[sample_cols].head(5).to_string())

        # 保存结果
        if args.output:
            merged_df.to_parquet(args.output, index=False)
            print(f"合并数据已保存到: {args.output}")
    else:
        print("真实API数据合并需要有效的API凭证和授权")
        print("请先使用 --mode=mock 测试合并逻辑")


def main():
    """主函数"""
    global args
    args = parse_args()

    # 设置日志
    setup_logging(args.config)
    logger = get_logger(__name__)

    # 加载配置
    config = load_config(args.config)

    print("=" * 60)
    print("抖音开放平台API演示")
    print("=" * 60)

    try:
        if args.mode == 'auth':
            demo_oauth_flow(config, args.code)

        elif args.mode == 'user':
            if not args.open_id:
                print("请提供 --open-id 参数")
                return
            demo_user_info(config, args.open_id)

        elif args.mode == 'video':
            if args.list_videos:
                if not args.open_id:
                    print("请提供 --open-id 参数")
                    return
                demo_list_videos(config, args.open_id)
            else:
                if not args.video_id:
                    print("请提供 --video-id 参数")
                    return
                demo_video_data(config, args.video_id)

        elif args.mode == 'merge':
            if not args.web_data:
                print("请提供 --web-data 参数")
                return
            demo_data_merge(config, args.web_data)

        elif args.mode == 'mock':
            print("Mock模式演示菜单:")
            print("  1. 用户信息演示")
            print("  2. 视频数据演示")
            print("  3. 视频列表演示")
            print("  4. 数据合并演示")
            print("  5. OAuth授权演示")

            choice = input("\n请选择演示项目 (1-5): ").strip()

            if choice == '1':
                demo_user_info(config, "mock_open_id_123")
            elif choice == '2':
                demo_video_data(config, "mock_video_123")
            elif choice == '3':
                demo_list_videos(config, "mock_open_id_123")
            elif choice == '4':
                # 使用样本数据
                sample_path = project_root / 'data/samples/sample_web_video_meta.csv'
                if sample_path.exists():
                    demo_data_merge(config, sample_path)
                else:
                    print(f"样本数据不存在: {sample_path}")
                    print("请先运行爬虫或使用 --web-data 参数指定数据文件")
            elif choice == '5':
                demo_oauth_flow(config, None)
            else:
                print("无效选择")

        else:
            print(f"未知模式: {args.mode}")
            print("可用模式: auth, user, video, merge, mock")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logger.error(f"演示脚本运行失败: {e}", exc_info=True)
        print(f"错误: {e}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == '__main__':
    main()