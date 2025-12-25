"""
YouTube Music Service
Fetches full songs from YouTube based on detected emotions
"""

import logging
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class YouTubeSong:
    """Represents a YouTube song"""
    video_id: str
    title: str
    artist: str
    duration: str  # e.g., "3:45"
    thumbnail: str
    emotion: str
    genre: str


# Curated playlists for each emotion - high quality songs
EMOTION_SONGS: Dict[str, List[Dict]] = {
    'happy': [
        {'video_id': 'ZbZSe6N_BXs', 'title': 'Happy', 'artist': 'Pharrell Williams', 'duration': '3:53'},
        {'video_id': 'ru0K8uYEZWw', 'title': "Can't Stop The Feeling", 'artist': 'Justin Timberlake', 'duration': '3:56'},
        {'video_id': 'OPf0YbXqDm0', 'title': 'Uptown Funk', 'artist': 'Bruno Mars', 'duration': '4:30'},
        {'video_id': 'fRh_vgS2dFE', 'title': "Sorry", 'artist': 'Justin Bieber', 'duration': '3:20'},
        {'video_id': 'JGwWNGJdvx8', 'title': 'Shape of You', 'artist': 'Ed Sheeran', 'duration': '4:24'},
        {'video_id': 'hT_nvWreIhg', 'title': 'Counting Stars', 'artist': 'OneRepublic', 'duration': '4:17'},
        {'video_id': 'PT2_F-1esPk', 'title': "Don't Start Now", 'artist': 'Dua Lipa', 'duration': '3:03'},
        {'video_id': 'kTJczUoc26U', 'title': 'Shake It Off', 'artist': 'Taylor Swift', 'duration': '3:39'},
        {'video_id': 'nfWlot6h_JM', 'title': 'Shut Up and Dance', 'artist': 'Walk The Moon', 'duration': '3:19'},
        {'video_id': 'IPXIgEAGe4U', 'title': 'I Gotta Feeling', 'artist': 'Black Eyed Peas', 'duration': '4:05'},
    ],
    'sad': [
        {'video_id': '4N3N1MlvVc4', 'title': 'Mad World', 'artist': 'Gary Jules', 'duration': '3:08'},
        {'video_id': 'RBumgq5yVrA', 'title': 'Someone Like You', 'artist': 'Adele', 'duration': '4:45'},
        {'video_id': 'hLQl3WQQoQ0', 'title': 'Someone You Loved', 'artist': 'Lewis Capaldi', 'duration': '3:02'},
        {'video_id': '450p7goxZqg', 'title': 'All Of Me', 'artist': 'John Legend', 'duration': '4:29'},
        {'video_id': 'YQHsXMglC9A', 'title': 'Hello', 'artist': 'Adele', 'duration': '4:55'},
        {'video_id': 'nVjsGKrE6E8', 'title': 'Fix You', 'artist': 'Coldplay', 'duration': '4:55'},
        {'video_id': 'JxPj3GAYYZ0', 'title': "It's So Hard To Say Goodbye", 'artist': 'Boyz II Men', 'duration': '3:25'},
        {'video_id': 'SpSMoBp8awM', 'title': 'Skinny Love', 'artist': 'Birdy', 'duration': '3:57'},
        {'video_id': 'WA4iX5D9Z64', 'title': 'Apologize', 'artist': 'OneRepublic', 'duration': '3:28'},
        {'video_id': 'LHCob76kigA', 'title': 'Say Something', 'artist': 'A Great Big World', 'duration': '3:49'},
    ],
    'angry': [
        {'video_id': '04F4xlWSFh0', 'title': 'In The End', 'artist': 'Linkin Park', 'duration': '3:36'},
        {'video_id': 'eVTXPUF4Oz4', 'title': 'Numb', 'artist': 'Linkin Park', 'duration': '3:06'},
        {'video_id': 'Hm7vnOC4hoY', 'title': 'Killing In The Name', 'artist': 'Rage Against The Machine', 'duration': '5:13'},
        {'video_id': '5abamRO41fE', 'title': 'Breaking The Habit', 'artist': 'Linkin Park', 'duration': '3:16'},
        {'video_id': 'CYqfRjwJrBY', 'title': 'Bring Me To Life', 'artist': 'Evanescence', 'duration': '3:58'},
        {'video_id': 'diYAc7gB-0A', 'title': 'Stronger', 'artist': 'Kanye West', 'duration': '5:12'},
        {'video_id': '09R8_2nJtjg', 'title': 'Sugar', 'artist': 'System Of A Down', 'duration': '2:36'},
        {'video_id': 'fGx6K90TmCI', 'title': "Bodies", 'artist': 'Drowning Pool', 'duration': '3:25'},
        {'video_id': 'L0MK7qz13bU', 'title': 'The Pretender', 'artist': 'Foo Fighters', 'duration': '4:29'},
        {'video_id': 'WM8bTdBs-cw', 'title': 'Psychosocial', 'artist': 'Slipknot', 'duration': '4:44'},
    ],
    'fear': [
        {'video_id': 'u9Dg-g7t2l4', 'title': 'Everybody Wants To Rule The World', 'artist': 'Tears For Fears', 'duration': '4:11'},
        {'video_id': 'Bm5iA4Zupek', 'title': 'Believer', 'artist': 'Imagine Dragons', 'duration': '3:24'},
        {'video_id': 'W-ao8qBGz4k', 'title': 'Bad Guy - Dark Ambient', 'artist': 'Various', 'duration': '3:30'},
        {'video_id': 'r7qovpFAGrQ', 'title': 'Dark Paradise', 'artist': 'Lana Del Rey', 'duration': '4:03'},
        {'video_id': 'k2qgadSvNyU', 'title': 'Haunted', 'artist': 'Taylor Swift', 'duration': '4:09'},
        {'video_id': 'g_uoH6hJilc', 'title': 'The Sound of Silence', 'artist': 'Disturbed', 'duration': '4:08'},
        {'video_id': 'eP4eqhWc7sI', 'title': 'Thriller', 'artist': 'Michael Jackson', 'duration': '5:57'},
        {'video_id': 'ZWir6wUkPtw', 'title': 'Bury A Friend', 'artist': 'Billie Eilish', 'duration': '3:13'},
        {'video_id': 'DyDfgMOUjCI', 'title': 'Bad Guy', 'artist': 'Billie Eilish', 'duration': '3:14'},
        {'video_id': 'qQkBeOisNM0', 'title': 'Sweet Dreams', 'artist': 'Eurythmics', 'duration': '3:36'},
    ],
    'surprise': [
        {'video_id': 'JRfuAukYTKg', 'title': 'Uptown Girl', 'artist': 'Billy Joel', 'duration': '3:17'},
        {'video_id': 'dQw4w9WgXcQ', 'title': 'Never Gonna Give You Up', 'artist': 'Rick Astley', 'duration': '3:33'},
        {'video_id': 'btPJPFnesV4', 'title': 'Eye of the Tiger', 'artist': 'Survivor', 'duration': '4:05'},
        {'video_id': '3JWTaaS7LdU', 'title': 'Take On Me', 'artist': 'a-ha', 'duration': '3:46'},
        {'video_id': '0J2QdDbelmY', 'title': 'Pompeii', 'artist': 'Bastille', 'duration': '3:34'},
        {'video_id': 'kXYiU_JCYtU', 'title': 'Numb Little Bug', 'artist': 'Em Beihold', 'duration': '2:53'},
        {'video_id': '60ItHLz5WEA', 'title': 'Africa', 'artist': 'Toto', 'duration': '4:55'},
        {'video_id': 'pXRviuL6vMY', 'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'duration': '5:55'},
        {'video_id': 'fJ9rUzIMcZQ', 'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'duration': '5:59'},
        {'video_id': 'hTWKbfoikeg', 'title': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'duration': '5:01'},
    ],
    'disgust': [
        {'video_id': 'o_1aF54DO60', 'title': 'Creep', 'artist': 'Radiohead', 'duration': '3:58'},
        {'video_id': 'lin-V6UYFRs', 'title': 'Radioactive', 'artist': 'Imagine Dragons', 'duration': '3:07'},
        {'video_id': 'RRKJiM9Njr8', 'title': 'Zombie', 'artist': 'Bad Wolves', 'duration': '4:16'},
        {'video_id': 'gH476CxJxfg', 'title': 'Monster', 'artist': 'Skillet', 'duration': '3:00'},
        {'video_id': '1mjlM_RnsVE', 'title': 'Enjoy The Silence', 'artist': 'Depeche Mode', 'duration': '4:17'},
        {'video_id': 'Epgo8ixX6Wo', 'title': 'Immigrant Song', 'artist': 'Led Zeppelin', 'duration': '2:26'},
        {'video_id': 'AOAtz8xWM0w', 'title': 'Animal I Have Become', 'artist': 'Three Days Grace', 'duration': '3:53'},
        {'video_id': 'tAGnKpE4NCI', 'title': 'Viva La Vida', 'artist': 'Coldplay', 'duration': '4:01'},
    ],
    'neutral': [
        {'video_id': 'YykjpeuMNEk', 'title': '22', 'artist': 'Taylor Swift', 'duration': '3:52'},
        {'video_id': 'pB-5XG-DbAA', 'title': 'Perfect', 'artist': 'Ed Sheeran', 'duration': '4:23'},
        {'video_id': 'bo_efYhYU2A', 'title': 'Sugar', 'artist': 'Maroon 5', 'duration': '3:55'},
        {'video_id': '7wtfhZwyrcc', 'title': 'Blinding Lights', 'artist': 'The Weeknd', 'duration': '3:20'},
        {'video_id': '60ItHLz5WEA', 'title': 'Africa', 'artist': 'Toto', 'duration': '4:55'},
        {'video_id': 'k4V3Mo61fJM', 'title': 'Let Her Go', 'artist': 'Passenger', 'duration': '4:12'},
        {'video_id': 'lp-EO5I60KA', 'title': 'Memories', 'artist': 'Maroon 5', 'duration': '3:09'},
        {'video_id': 'QDYfEBY9NM4', 'title': 'Let It Be', 'artist': 'The Beatles', 'duration': '4:03'},
        {'video_id': 'hLQl3WQQoQ0', 'title': 'Before You Go', 'artist': 'Lewis Capaldi', 'duration': '3:35'},
        {'video_id': 'kJQP7kiw5Fk', 'title': 'Despacito', 'artist': 'Luis Fonsi', 'duration': '4:41'},
    ],
    'calm': [
        {'video_id': 'HEXWRTEbj1I', 'title': 'Wake Me Up When September Ends', 'artist': 'Green Day', 'duration': '4:45'},
        {'video_id': 'HcVv9R1ZR84', 'title': "River Flows In You", 'artist': 'Yiruma', 'duration': '3:41'},
        {'video_id': 'k4V3Mo61fJM', 'title': 'Let Her Go', 'artist': 'Passenger', 'duration': '4:12'},
        {'video_id': 'RgKAFK5djSk', 'title': 'See You Again', 'artist': 'Wiz Khalifa ft. Charlie Puth', 'duration': '3:58'},
        {'video_id': '1k8craCGpgs', 'title': "I Don't Want To Miss A Thing", 'artist': 'Aerosmith', 'duration': '4:59'},
        {'video_id': 'nSDgHBxUbVQ', 'title': "Photograph", 'artist': 'Ed Sheeran', 'duration': '4:18'},
        {'video_id': 'SlPhMPnQ58k', 'title': 'Watermark', 'artist': 'Enya', 'duration': '2:25'},
        {'video_id': '2bosouX_d8Y', 'title': 'Only Time', 'artist': 'Enya', 'duration': '3:38'},
        {'video_id': 'YR5ApYxkU-U', 'title': 'Moonlight Sonata', 'artist': 'Beethoven', 'duration': '5:31'},
        {'video_id': 'raNGeq3_DtM', 'title': 'Canon in D', 'artist': 'Pachelbel', 'duration': '5:17'},
    ]
}

# Genre mapping for emotions
EMOTION_GENRES = {
    'happy': ['pop', 'dance', 'upbeat'],
    'sad': ['ballad', 'acoustic', 'piano'],
    'angry': ['rock', 'metal', 'intense'],
    'fear': ['dark', 'ambient', 'suspense'],
    'surprise': ['classic', 'eclectic', 'energetic'],
    'disgust': ['alternative', 'grunge', 'experimental'],
    'neutral': ['pop', 'easy listening', 'mainstream'],
    'calm': ['acoustic', 'piano', 'ambient', 'classical']
}


class YouTubeMusicService:
    """Service for fetching YouTube songs based on emotions"""
    
    def __init__(self):
        self.songs = EMOTION_SONGS
        self.genres = EMOTION_GENRES
        self._history: List[str] = []  # Track recently played to avoid repeats
        
    def get_song_for_emotion(self, emotion: str, avoid_recent: bool = True) -> YouTubeSong:
        """
        Get a random song matching the emotion
        
        Args:
            emotion: Detected emotion
            avoid_recent: Whether to avoid recently played songs
            
        Returns:
            YouTubeSong with video details
        """
        emotion_lower = emotion.lower()
        
        # Fall back to neutral if emotion not found
        if emotion_lower not in self.songs:
            logger.warning(f"Unknown emotion '{emotion}', falling back to neutral")
            emotion_lower = 'neutral'
        
        available_songs = self.songs[emotion_lower]
        
        # Filter out recently played if requested
        if avoid_recent and len(self._history) > 0:
            filtered = [s for s in available_songs if s['video_id'] not in self._history]
            if filtered:
                available_songs = filtered
            else:
                # Clear history if all songs have been played
                self._history.clear()
        
        # Select random song
        song_data = random.choice(available_songs)
        
        # Add to history
        self._history.append(song_data['video_id'])
        if len(self._history) > 5:
            self._history.pop(0)
        
        # Create YouTubeSong object
        return YouTubeSong(
            video_id=song_data['video_id'],
            title=song_data['title'],
            artist=song_data['artist'],
            duration=song_data['duration'],
            thumbnail=f"https://img.youtube.com/vi/{song_data['video_id']}/hqdefault.jpg",
            emotion=emotion_lower,
            genre=random.choice(self.genres.get(emotion_lower, ['pop']))
        )
    
    def get_multiple_songs(self, emotion: str, count: int = 5) -> List[YouTubeSong]:
        """Get multiple songs for a playlist"""
        emotion_lower = emotion.lower()
        if emotion_lower not in self.songs:
            emotion_lower = 'neutral'
            
        available = self.songs[emotion_lower]
        selected = random.sample(available, min(count, len(available)))
        
        return [
            YouTubeSong(
                video_id=s['video_id'],
                title=s['title'],
                artist=s['artist'],
                duration=s['duration'],
                thumbnail=f"https://img.youtube.com/vi/{s['video_id']}/hqdefault.jpg",
                emotion=emotion_lower,
                genre=random.choice(self.genres.get(emotion_lower, ['pop']))
            )
            for s in selected
        ]
    
    def get_search_query(self, emotion: str) -> str:
        """Generate a YouTube search query for the emotion"""
        queries = {
            'happy': 'happy upbeat songs playlist',
            'sad': 'sad emotional songs playlist',
            'angry': 'intense rock metal songs',
            'fear': 'dark atmospheric music',
            'surprise': 'epic dramatic songs',
            'disgust': 'alternative grunge music',
            'neutral': 'popular hits playlist',
            'calm': 'relaxing peaceful music'
        }
        return queries.get(emotion.lower(), 'popular music playlist')


# Singleton instance
_youtube_service: Optional[YouTubeMusicService] = None

def get_youtube_service() -> YouTubeMusicService:
    """Get or create YouTube music service instance"""
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeMusicService()
    return _youtube_service


def get_song_for_emotion(emotion: str) -> Dict:
    """
    Get a YouTube song for the given emotion
    
    Args:
        emotion: Detected emotion
        
    Returns:
        Dictionary with song details
    """
    service = get_youtube_service()
    song = service.get_song_for_emotion(emotion)
    
    return {
        'video_id': song.video_id,
        'title': song.title,
        'artist': song.artist,
        'duration': song.duration,
        'thumbnail': song.thumbnail,
        'emotion': song.emotion,
        'genre': song.genre,
        'embed_url': f'https://www.youtube.com/embed/{song.video_id}?autoplay=1&rel=0',
        'watch_url': f'https://www.youtube.com/watch?v={song.video_id}'
    }


def get_playlist_for_emotion(emotion: str, count: int = 5) -> List[Dict]:
    """Get multiple songs as a playlist"""
    service = get_youtube_service()
    songs = service.get_multiple_songs(emotion, count)
    
    return [
        {
            'video_id': s.video_id,
            'title': s.title,
            'artist': s.artist,
            'duration': s.duration,
            'thumbnail': s.thumbnail,
            'emotion': s.emotion,
            'genre': s.genre,
            'embed_url': f'https://www.youtube.com/embed/{s.video_id}?autoplay=1&rel=0',
            'watch_url': f'https://www.youtube.com/watch?v={s.video_id}'
        }
        for s in songs
    ]
