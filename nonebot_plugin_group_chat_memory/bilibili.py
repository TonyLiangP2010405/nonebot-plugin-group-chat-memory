from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from .api import ChatClient

logger = logging.getLogger(__name__)

HOME_URL = "https://www.bilibili.com/"
NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
WBI_SEARCH_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SEARCH_TIMEOUT = 8.0
MAX_RESULTS = 5
MIXIN_KEY_TTL = 1800.0

# B站 wbi 签名使用的固定打乱表 见 bilibili-API-collect
MIXIN_KEY_ORDER = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)
UNSAFE_QUERY_CHARS = ("!", "'", "(", ")", "*")

DECISION_SYSTEM_PROMPT = (
    "你是群聊机器人「馒头」的回复助理，负责决定这条回复要不要再配一个 B站视频。\n"
    "需要的情况：群友明确想看视频、求番剧/动漫/游戏/整活视频推荐，"
    "或者在聊某个具体的作品、角色、主播并很可能想看点相关的视频。\n"
    "不需要的情况：日常寒暄、接梗吐槽、情绪安慰、正经的技术/学习/健康/安全问题，"
    "或者想不出能搜到高相关视频的关键词。\n"
    "只输出一行 JSON，不要解释，不要 Markdown：\n"
    '需要时输出 {"search": true, "keyword": "适合在B站搜索的简短关键词"}\n'
    '不需要时输出 {"search": false}'
)


class BilibiliError(RuntimeError):
    """B站搜索失败；异常消息只包含状态码或原因，不含视频正文。"""


@dataclass(frozen=True)
class BilibiliVideo:
    """一条 B站搜索结果，字段都来自搜索接口，缺省值已归一。"""

    title: str
    author: str
    url: str
    cover: str
    play: int
    like: int
    review: int


@dataclass(frozen=True)
class VideoCue:
    """模型判断需要配视频时给出的搜索关键词。"""

    keyword: str


def video_card_text(video: BilibiliVideo) -> str:
    """按群里的分享样式拼出视频卡片文字：标题、UP、数据、链接。"""
    return (
        f"🎬 {video.title}\n"
        f"👤 UP: {video.author}\n"
        f"▶ 播放: {video.play:,} 👍 点赞: {video.like:,} 💬 评论: {video.review:,}\n"
        f"🔗 {video.url}"
    )


def build_decision_messages(current_text: str, answer: str) -> list[dict[str, str]]:
    """组装判断是否需要视频的 messages，只带当前消息和准备发出的回复。"""
    return [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"群友说：{current_text.strip()}\n馒头准备回复：{answer.strip()}",
        },
    ]


def parse_decision(content: str) -> VideoCue | None:
    """解析模型输出的 JSON；不需要、格式不对或关键词不可用时返回 None。"""
    payload = _extract_json_object(content)
    if not isinstance(payload, dict):
        return None
    if payload.get("search") is not True:
        return None
    keyword = str(payload.get("keyword") or "").strip()
    if not keyword or len(keyword) > 60:
        return None
    return VideoCue(keyword=keyword)


async def decide_video(
    client: ChatClient, current_text: str, answer: str
) -> VideoCue | None:
    """让聊天模型自己判断是否需要视频；调用失败按“不需要”处理。"""
    try:
        content = await client.complete_prompt(
            build_decision_messages(current_text, answer)
        )
    except Exception as exc:
        logger.warning("B站视频判断失败：%s", type(exc).__name__)
        return None
    return parse_decision(content)


class BilibiliClient:
    """B站视频搜索：优先免签名接口，被风控时回退到 wbi 签名接口。"""

    def __init__(self, *, timeout: float = SEARCH_TIMEOUT) -> None:
        self._timeout = timeout
        self._cookies: dict[str, str] = {}
        self._mixin_key = ""
        self._mixin_expires = 0.0

    async def search(self, keyword: str) -> BilibiliVideo | None:
        """搜索关键词并返回第一条视频；没有结果返回 None，接口异常抛 BilibiliError。"""
        keyword = keyword.strip()
        if not keyword:
            return None
        videos = await self._search(keyword)
        return videos[0] if videos else None

    async def _search(self, keyword: str) -> list[BilibiliVideo]:
        failure: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout, headers=_search_headers()
        ) as client:
            for attempt in (self._search_plain, self._search_wbi):
                try:
                    videos = await attempt(client, keyword)
                except (httpx.HTTPError, BilibiliError, ValueError) as exc:
                    failure = exc
                    logger.debug("B站搜索（%s）失败：%r", attempt.__name__, exc)
                    continue
                if videos:
                    return videos
        if failure is not None:
            raise BilibiliError("B站搜索失败。") from failure
        return []

    async def _search_plain(
        self, client: httpx.AsyncClient, keyword: str
    ) -> list[BilibiliVideo]:
        response = await client.get(
            SEARCH_URL,
            params={"search_type": "video", "keyword": keyword, "page": 1},
        )
        return _parse_payload(response)

    async def _search_wbi(
        self, client: httpx.AsyncClient, keyword: str
    ) -> list[BilibiliVideo]:
        await self._ensure_session(client)
        params = sign_params(
            {"search_type": "video", "keyword": keyword, "page": 1}, self._mixin_key
        )
        response = await client.get(WBI_SEARCH_URL, params=params)
        return _parse_payload(response)

    async def _ensure_session(self, client: httpx.AsyncClient) -> None:
        """补齐 buvid cookie 与 wbi 密钥；cookie 与密钥各自缓存，密钥会过期刷新。"""
        if not self._cookies:
            try:
                await client.get(HOME_URL)
            except httpx.HTTPError as exc:
                logger.debug("B站首页请求失败：%r", exc)
            else:
                self._cookies = {
                    cookie.name: cookie.value for cookie in client.cookies.jar
                }
                client.cookies.update(self._cookies)
        if self._mixin_key and time.monotonic() < self._mixin_expires:
            return
        response = await client.get(NAV_URL)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BilibiliError("B站 wbi 密钥响应不是合法 JSON。") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        wbi_img = data.get("wbi_img") if isinstance(data, dict) else None
        img_key = _key_from_url(wbi_img.get("img_url")) if isinstance(wbi_img, dict) else ""
        sub_key = _key_from_url(wbi_img.get("sub_url")) if isinstance(wbi_img, dict) else ""
        if not img_key or not sub_key:
            raise BilibiliError("B站 wbi 密钥缺失。")
        self._mixin_key = _mixin_key(img_key, sub_key)
        self._mixin_expires = time.monotonic() + MIXIN_KEY_TTL


def sign_params(
    params: dict[str, Any], mixin_key: str, *, timestamp: int | None = None
) -> dict[str, str]:
    """按 wbi 规则签名：过滤特殊字符、按 key 排序、附 wts，再算 md5 w_rid。"""
    signed = {key: _filter_value(value) for key, value in params.items()}
    signed["wts"] = str(int(time.time() if timestamp is None else timestamp))
    query = "&".join(
        f"{quote(key, safe='')}={quote(signed[key], safe='')}" for key in sorted(signed)
    )
    signed["w_rid"] = hashlib.md5(f"{query}{mixin_key}".encode()).hexdigest()
    return signed


def _mixin_key(img_key: str, sub_key: str) -> str:
    source = img_key + sub_key
    if len(source) < len(MIXIN_KEY_ORDER):
        raise BilibiliError("B站 wbi 密钥长度异常。")
    return "".join(source[index] for index in MIXIN_KEY_ORDER)[:32]


def _key_from_url(value: Any) -> str:
    filename = str(value or "").rsplit("/", 1)[-1]
    return filename.split(".")[0]


def _filter_value(value: Any) -> str:
    text = str(value)
    for char in UNSAFE_QUERY_CHARS:
        text = text.replace(char, "")
    return text


def _search_headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Referer": HOME_URL,
        "Accept": "application/json, text/plain, */*",
    }


def _parse_payload(response: httpx.Response) -> list[BilibiliVideo]:
    if response.status_code != 200:
        raise BilibiliError(f"B站搜索返回 HTTP {response.status_code}。")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BilibiliError("B站搜索响应不是合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise BilibiliError("B站搜索响应格式异常。")
    if payload.get("code") != 0:
        raise BilibiliError(f"B站搜索返回 code={payload.get('code')}。")
    return _parse_result(payload.get("data"))


def _parse_result(data: Any) -> list[BilibiliVideo]:
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, list):
        return []
    videos: list[BilibiliVideo] = []
    for item in result:
        video = _parse_item(item)
        if video is None:
            continue
        videos.append(video)
        if len(videos) >= MAX_RESULTS:
            break
    return videos


def _parse_item(item: Any) -> BilibiliVideo | None:
    if not isinstance(item, dict) or item.get("type") != "video":
        return None
    bvid = str(item.get("bvid") or "").strip()
    title = _plain_text(item.get("title"))
    if not bvid or not title:
        return None
    return BilibiliVideo(
        title=title,
        author=_plain_text(item.get("author")) or "未知 UP 主",
        url=f"https://www.bilibili.com/video/{bvid}",
        cover=_image_url(item.get("pic")),
        play=_count(item.get("play")),
        like=_count(item.get("like")),
        review=_count(item.get("review")),
    )


def _plain_text(value: Any) -> str:
    """搜索结果标题带 <em class="keyword"> 高亮标签，这里去掉标签并还原实体。"""
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return html.unescape(text).strip()


def _image_url(value: Any) -> str:
    url = str(value or "").strip()
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("http://"):
        return f"https://{url[len('http://') :]}"
    return url if url.startswith("https://") else ""


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _extract_json_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None
