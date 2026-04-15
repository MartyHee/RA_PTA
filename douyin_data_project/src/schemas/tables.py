"""
Data schemas for Douyin data project.

This module defines Pydantic models for the data tables described in
`docs/Data_description.md`. These models are used for data validation,
serialization, and documentation.

If Pydantic is not installed, these classes can be used as type hints
with dataclass-like structure.
"""
from datetime import datetime, date
from typing import List, Optional, Union
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback base class
    class BaseModel:
        """Fallback BaseModel when Pydantic is not available."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def Field(default=None, **kwargs):
        """Mock Field for type hints."""
        return default

    def validator(*args, **kwargs):
        """Mock validator decorator."""
        def decorator(func):
            return func
        return decorator

# Common validators and utilities
def normalize_count(value: Optional[str]) -> Optional[int]:
    """Normalize count strings like '1.2w' or '5k' to integers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if 'w' in value:
            num = float(value.replace('w', ''))
            return int(num * 10000)
        elif 'k' in value:
            num = float(value.replace('k', ''))
            return int(num * 1000)
        elif value.isdigit():
            return int(value)
    return None


class RawWebVideoData(BaseModel):
    """Raw web video data layer.

    Used for保存网页抓取过程中的原始信息，强调追溯性与调试能力。
    Corresponds to `raw_web_video_data` in Data_description.md.
    """
    if PYDANTIC_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    crawl_id: str = Field(..., description="本次抓取任务ID")
    source_entry: str = Field(..., description="数据入口类型，如 search/topic/rank/manual_url")
    original_url: Optional[str] = Field(None, description="原始输入URL")
    canonical_video_url: Optional[str] = Field(None, description="标准视频详情页URL")
    fetched_url: Optional[str] = Field(None, description="实际抓取的URL")
    page_type: Optional[str] = Field(None, description="页面类型")
    page_url: str = Field(..., description="视频页面URL")
    raw_html_path: Optional[Path] = Field(None, description="原始HTML保存路径")
    rendered_html_path: Optional[Path] = Field(None, description="渲染后的HTML保存路径（浏览器模式）")
    raw_json_blob: Optional[str] = Field(None, description="从页面脚本中解析出的原始JSON片段")
    http_status: int = Field(..., description="请求状态码")
    crawl_time: datetime = Field(..., description="抓取时间")
    parse_status: str = Field(..., description="解析状态，如 success/partial_success/fail")
    parse_error_msg: Optional[str] = Field(None, description="解析异常信息")

    @validator('source_entry')
    def validate_source_entry(cls, v):
        allowed = {'search', 'topic', 'rank', 'manual_url'}
        if v not in allowed:
            raise ValueError(f'source_entry must be one of {allowed}')
        return v

    @validator('parse_status')
    def validate_parse_status(cls, v):
        allowed = {'success', 'partial_success', 'fail'}
        if v not in allowed:
            raise ValueError(f'parse_status must be one of {allowed}')
        return v

    @validator('page_type')
    def validate_page_type(cls, v):
        if v is None:
            return v
        allowed = {'video_detail', 'jingxuan_modal', 'jingxuan_feed', 'unknown'}
        if v not in allowed:
            raise ValueError(f'page_type must be one of {allowed}')
        return v


class WebVideoMeta(BaseModel):
    """Web video metadata table.

    网页端采集后的主业务表，用于后续清洗、分析与与API合并。
    Corresponds to `web_video_meta` in Data_description.md.
    """
    if PYDANTIC_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: str = Field(..., description="视频唯一标识")
    page_url: str = Field(..., description="视频页面URL")
    author_id: Optional[str] = Field(None, description="作者ID，若页面可提取")
    author_name: Optional[str] = Field(None, description="作者昵称")
    author_profile_url: Optional[str] = Field(None, description="作者主页URL")
    desc_text: Optional[str] = Field(None, description="视频文案/标题文本，统一为单字段")
    publish_time_raw: Optional[str] = Field(None, description="页面显示的原始发布时间文本")
    publish_time_std: Optional[datetime] = Field(None, description="标准化后的发布时间")
    like_count_raw: Optional[str] = Field(None, description="页面原始点赞数文本，如 '1.2w'")
    comment_count_raw: Optional[str] = Field(None, description="页面原始评论数文本")
    share_count_raw: Optional[str] = Field(None, description="页面原始分享数文本")
    like_count: Optional[int] = Field(None, description="归一化后的点赞数")
    comment_count: Optional[int] = Field(None, description="归一化后的评论数")
    share_count: Optional[int] = Field(None, description="归一化后的分享数")
    collect_count: Optional[int] = Field(None, description="收藏数，网页端不稳定，仅作为可选字段")
    hashtag_list: Optional[List[str]] = Field(default_factory=list, description="从文案中抽取的 #话题 列表")
    hashtag_count: int = Field(0, description="话题数量，无则为0")
    cover_url: Optional[str] = Field(None, description="封面图片链接")
    music_name: Optional[str] = Field(None, description="关联音乐名称，若页面可提取")
    duration_sec: Optional[int] = Field(None, description="视频时长（秒），若页面可提取")
    source_entry: str = Field(..., description="采样来源")
    crawl_time: datetime = Field(..., description="抓取时间")

    @validator('hashtag_count', always=True)
    def set_hashtag_count(cls, v, values):
        """Set hashtag_count based on hashtag_list if not provided."""
        if 'hashtag_list' in values and values['hashtag_list'] is not None:
            return len(values['hashtag_list'])
        return v or 0

    @validator('like_count', pre=True)
    def normalize_like_count(cls, v, values):
        if v is not None:
            return v
        if 'like_count_raw' in values and values['like_count_raw']:
            return normalize_count(values['like_count_raw'])
        return None

    @validator('comment_count', pre=True)
    def normalize_comment_count(cls, v, values):
        if v is not None:
            return v
        if 'comment_count_raw' in values and values['comment_count_raw']:
            return normalize_count(values['comment_count_raw'])
        return None

    @validator('share_count', pre=True)
    def normalize_share_count(cls, v, values):
        if v is not None:
            return v
        if 'share_count_raw' in values and values['share_count_raw']:
            return normalize_count(values['share_count_raw'])
        return None


class ProcessedVideoData(BaseModel):
    """Processed video data table.

    推荐实验前的数据准备表，字段建立在网页端稳定可用信息基础之上。
    Corresponds to `processed_video_data` in Data_description.md.
    """
    video_id: str = Field(..., description="视频唯一标识")
    author_id: Optional[str] = Field(None, description="作者ID")
    desc_clean: Optional[str] = Field(None, description="清洗后的文案文本")
    text_length: int = Field(..., description="文案长度")
    publish_date: Optional[date] = Field(None, description="发布日期")
    publish_hour: Optional[int] = Field(None, description="发布时间小时", ge=0, le=23)
    publish_weekday: Optional[int] = Field(None, description="星期几", ge=0, le=6)
    is_weekend: Optional[int] = Field(None, description="是否周末", ge=0, le=1)
    hashtag_list: Optional[List[str]] = Field(default_factory=list, description="标准化后的话题列表")
    hashtag_count: int = Field(..., description="话题数量")
    like_count: Optional[int] = Field(None, description="点赞数")
    comment_count: Optional[int] = Field(None, description="评论数")
    share_count: Optional[int] = Field(None, description="分享数")
    collect_count: Optional[int] = Field(None, description="收藏数")
    engagement_score: Optional[float] = Field(None, description="互动分数")
    source_entry: str = Field(..., description="采样入口")
    crawl_date: date = Field(..., description="抓取日期")
    data_version: str = Field(..., description="数据版本号")

    @validator('engagement_score', pre=True)
    def calculate_engagement_score(cls, v, values):
        """Calculate engagement score if not provided."""
        if v is not None:
            return v

        weights = {'like': 1.0, 'comment': 2.0, 'share': 3.0}
        score = 0.0

        if 'like_count' in values and values['like_count']:
            score += weights['like'] * values['like_count']
        if 'comment_count' in values and values['comment_count']:
            score += weights['comment'] * values['comment_count']
        if 'share_count' in values and values['share_count']:
            score += weights['share'] * values['share_count']

        return score if score > 0 else None

    @validator('text_length', pre=True)
    def calculate_text_length(cls, v, values):
        """Calculate text length from desc_clean if not provided."""
        if v is not None:
            return v
        if 'desc_clean' in values and values['desc_clean']:
            return len(values['desc_clean'])
        return 0


class AuthorDim(BaseModel):
    """Author dimension table.

    作者维表在网页端只保留最小可行信息，避免依赖不稳定字段。
    Corresponds to `author_dim` in Data_description.md.
    """
    author_id: str = Field(..., description="作者唯一标识")
    author_name_latest: Optional[str] = Field(None, description="最近一次采集到的昵称")
    author_profile_url: Optional[str] = Field(None, description="作者主页URL")
    sampled_video_count: int = Field(..., description="当前样本中该作者的视频数量", ge=1)
    first_seen: datetime = Field(..., description="首次出现时间")
    last_seen: datetime = Field(..., description="最后一次出现时间")


class ApiVideoStats(BaseModel):
    """API video statistics table.

    API端用于补充网页端难以稳定获取的视频统计信息。
    Corresponds to `api_video_stats` in Data_description.md.
    """
    video_id: str = Field(..., description="视频ID / item_id")
    open_id: str = Field(..., description="授权用户标识")
    stat_time: datetime = Field(..., description="统计拉取时间")
    play_count: Optional[int] = Field(None, description="播放数")
    digg_count: Optional[int] = Field(None, description="点赞数")
    comment_count: Optional[int] = Field(None, description="评论数")
    share_count: Optional[int] = Field(None, description="分享数")
    cover_url: Optional[str] = Field(None, description="封面图URL")
    create_time: Optional[datetime] = Field(None, description="视频创建时间")
    api_pull_time: datetime = Field(..., description="API实际拉取时间")


class ApiUserProfile(BaseModel):
    """API user profile table.

    授权用户公开信息表。
    Corresponds to `api_user_profile` in Data_description.md.
    """
    open_id: str = Field(..., description="授权用户标识")
    nickname: Optional[str] = Field(None, description="用户昵称")
    avatar_url: Optional[str] = Field(None, description="用户头像")
    gender: Optional[str] = Field(None, description="性别")
    province: Optional[str] = Field(None, description="省份/地区")
    city: Optional[str] = Field(None, description="城市/地区")
    pull_time: datetime = Field(..., description="拉取时间")

    @validator('gender')
    def validate_gender(cls, v):
        if v is None:
            return v
        allowed = {'male', 'female', 'unknown'}
        if v.lower() not in allowed:
            raise ValueError(f'gender must be one of {allowed}')
        return v.lower()


# Export all schemas
__all__ = [
    'RawWebVideoData',
    'WebVideoMeta',
    'ProcessedVideoData',
    'AuthorDim',
    'ApiVideoStats',
    'ApiUserProfile',
    'normalize_count',
]