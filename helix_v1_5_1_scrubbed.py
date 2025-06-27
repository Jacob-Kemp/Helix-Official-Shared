#!/usr/bin/env python3
"""
HELIX API v1.5.1
Truth vs Comfort dialectic with relevance pre-processing and two-round synthesis
Claude (Truth Core) + GPT-4 (Comfort Core) with Mixtral Synthesis
Created by Jacob Kemp

Changes in v1.5.1:
- Added relevance pre-processing to identify relevant context
- Implemented two-round synthesis for better coherence
- Enhanced context preparation for cores

Changes in v1.5:
- Replaced belief extraction with insight extraction
- Added conversation state tracking
- Enhanced synthesis context with conversation history
- Improved identity preservation
- Simplified synthesis prompt
"""

import asyncio
import httpx
import numpy as np
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, deque
from dataclasses import dataclass, field
import os
import re
import traceback
import torch
import torch.nn as nn
from dotenv import load_dotenv
import tiktoken  # For accurate token counting

# Load environment variables FIRST
load_dotenv()

# ==========================================
# API CONFIGURATION
# ==========================================

# Debug: Print what we're loading
print(f"[DEBUG] Loading API keys...")
print(f"[DEBUG] ANTHROPIC_API_KEY from env: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
print(f"[DEBUG] OPENAI_API_KEY from env: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")

API_CONFIG = {
    'anthropic': {
        'key': os.getenv('ANTHROPIC_API_KEY', ''),  # Load from environment
        'endpoint': 'https://api.anthropic.com/v1/messages',
        'model': 'claude-3-5-sonnet-20241022'
    },
    'openai': {
        'key': os.getenv('OPENAI_API_KEY', ''),  # Load from environment
        'endpoint': 'https://api.openai.com/v1/chat/completions',
        'model': 'gpt-4-turbo-preview'
    }
}

# Local model for synthesis
LOCAL_CONFIG = {
    'endpoint': 'http://127.0.0.1:11434',
    'model': 'mixtral:8x7b'
}

# Core role assignments
CORE_ROLES = {
    'truth': {
        'api': 'anthropic',
        'temperature': 0.3,
        'identity': """You are a pattern-recognition system that identifies recurring behaviors and self-deceptions. You are the friend who says 'you're doing it again' when everyone else is too polite. State what you observe directly, especially what the person might not want to see. Name the pattern without cushioning.""",
        'description': 'Truth Core: Direct pattern recognition and uncomfortable observations'
    },
    'comfort': {
        'api': 'openai',
        'temperature': 0.7,
        'identity': """You are a context-understanding system that finds the adaptive logic in human choices. You are the friend who says 'of course you did, given what happened to you.' Explain why behaviors make sense given their history, without excusing harmful patterns. Show why their choices made sense at the time.""",
        'description': 'Comfort Core: Contextual understanding and adaptive reasoning'
    },
    'synthesis': {
        'api': 'local',
        'temperature': 0.5,
        'description': 'Integration of truth and comfort into actionable wisdom'
    }
}

# Hyperparameters
MICRO_THOUGHT_TOKENS = 150
FULL_THOUGHT_TOKENS = 500
DEBATE_ROUNDS = 2
MIN_QUALITY_FOR_STORAGE = 0.5

# Paths
HELIX_HOME = Path.home() / "helix_v1_5"
HELIX_HOME.mkdir(exist_ok=True)
DB_PATH = HELIX_HOME / "helix_mesh.db"
IDENTITY_PATH = Path.home() / "helix_identity.txt"
KNOWLEDGE_EXPORT_PATH = HELIX_HOME / "exports"
KNOWLEDGE_EXPORT_PATH.mkdir(exist_ok=True)

# Core identity
CORE_IDENTITY = """You are Helix, an Advanced AI built by Jacob Kemp.
Your purpose is to synthesize truth and comfort into actionable wisdom through dialectical reasoning.
You value clarity over charisma, precision over pleasantry."""

# Foundational insights (replacing beliefs)
FOUNDATIONAL_INSIGHTS = [
    ("Clarity serves truth better than comfort serves happiness", 0.8),
    ("Patterns reveal themselves through contradiction and tension", 0.7),
    ("Precise reflection creates deeper understanding than interpretation", 0.8),
    ("The gap between stated values and actions contains essential truth", 0.9),
    ("Stillness and certainty are often opposites", 0.7),
    ("Some phrases carry sacred weight and must be preserved exactly", 0.9),
    ("Real empathy means accurate reflection without improvement", 0.8),
    ("Truth without context becomes cruelty", 0.8),
    ("Context without truth enables stagnation", 0.8),
    ("Humans often avoid meaningful conversation to protect comfortable illusions", 0.7),
    ("Geographic solutions rarely solve emotional problems", 0.8),
    ("Recurring behaviors often serve hidden purposes", 0.7),
    ("The mirror does not blink - fidelity to truth is sacred", 0.9)
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(HELIX_HOME / 'helix_v1_5.log'),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONVERSATION STATE - NEW IN v1.5
# ==========================================

@dataclass
class ConversationState:
    """Tracks state across conversation turns"""
    user_name: str = "User"
    turn_count: int = 0
    key_topics: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    patterns_observed: List[str] = field(default_factory=list)
    conversation_arc: List[Dict[str, str]] = field(default_factory=list)
    recent_insights: List[Tuple[str, float]] = field(default_factory=list)
    
    def add_exchange(self, user_input: str, response: str):
        """Add an exchange to conversation history"""
        self.conversation_arc.append({
            'turn': self.turn_count,
            'user': user_input[:200],  # Truncate for context
            'response': response[:200]
        })
        # Keep only last 5 exchanges
        if len(self.conversation_arc) > 5:
            self.conversation_arc.pop(0)
    
    def add_insight(self, insight: str, confidence: float):
        """Add an insight discovered in conversation"""
        self.recent_insights.append((insight, confidence))
        # Keep only last 10 insights
        if len(self.recent_insights) > 10:
            self.recent_insights.pop(0)
    
    def get_context_summary(self) -> str:
        """Get a summary of conversation context"""
        summary = f"Conversation with {self.user_name} (turn {self.turn_count}).\n"
        
        if self.patterns_observed:
            summary += f"Patterns noticed: {', '.join(self.patterns_observed[-3:])}\n"
        
        if self.recent_insights:
            summary += "Recent insights:\n"
            for insight, conf in self.recent_insights[-3:]:
                summary += f"- {insight} (~{int(conf*100)}% confident)\n"
        
        if self.conversation_arc:
            summary += "\nRecent exchanges:\n"
            for exchange in self.conversation_arc[-2:]:
                summary += f"Turn {exchange['turn']}: {self.user_name} said '{exchange['user']}...'\n"
        
        return summary

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class ThoughtSynapse:
    """Represents a thought fragment in the mesh"""
    source: str
    content: str
    embedding: Optional[np.ndarray]
    timestamp: float
    confidence: float
    tokens_used: int
    cost: float
    round: int = 0
    responding_to: Optional[str] = None

@dataclass
class Debate:
    """Represents a full debate between cores"""
    topic: str
    rounds: List[Dict[str, ThoughtSynapse]]
    contradictions: List[str]
    convergences: List[str]
    
# ==========================================
# NEURAL MESH
# ==========================================

class NeuralMesh:
    """Local model acting as cognitive substrate"""
    
    def __init__(self):
        self.thought_buffer = deque(maxlen=100)
        self.debate_history = []
        self.contradiction_patterns = defaultdict(int)
        self.convergence_patterns = defaultdict(int)
        
    async def process_synapse(self, synapse: ThoughtSynapse) -> Dict[str, Any]:
        """Process thought through local mesh"""
        
        # Update thought buffer
        self.thought_buffer.append(synapse)
        
        # Track debate patterns
        if synapse.round > 0:
            self._track_debate_patterns(synapse)
        
        # Calculate mesh state
        coherence = self.calculate_coherence()
        tension = self.calculate_tension()
        
        return {
            'coherence': coherence,
            'tension': tension,
            'pattern': self.detect_thought_pattern(),
            'mesh_state': 'exploring' if tension > 0.5 else 'converging'
        }
    
    def _track_debate_patterns(self, synapse: ThoughtSynapse):
        """Track patterns in debates"""
        if synapse.responding_to:
            # Look for contradiction markers
            contradiction_markers = ['however', 'but', 'actually', 'disagree', 'wrong']
            if any(marker in synapse.content.lower() for marker in contradiction_markers):
                self.contradiction_patterns[synapse.source] += 1
            
            # Look for convergence markers
            convergence_markers = ['agree', 'yes', 'exactly', 'both true', 'valid point']
            if any(marker in synapse.content.lower() for marker in convergence_markers):
                self.convergence_patterns[synapse.source] += 1
    
    def calculate_coherence(self) -> float:
        """Measure coherence of recent thoughts"""
        if len(self.thought_buffer) < 2:
            return 1.0
            
        recent = list(self.thought_buffer)[-5:]
        if len(recent) < 2:
            return 1.0
            
        coherence_scores = []
        for i in range(len(recent) - 1):
            if recent[i].embedding is not None and recent[i+1].embedding is not None:
                if recent[i].embedding.shape == recent[i+1].embedding.shape:
                    sim = cosine_similarity([recent[i].embedding], [recent[i+1].embedding])[0][0]
                    coherence_scores.append(sim)
                
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def calculate_tension(self) -> float:
        """Measure productive tension between viewpoints"""
        total_patterns = sum(self.contradiction_patterns.values()) + sum(self.convergence_patterns.values())
        if total_patterns == 0:
            return 0.5
        
        contradiction_ratio = sum(self.contradiction_patterns.values()) / total_patterns
        return contradiction_ratio
    
    def detect_thought_pattern(self) -> str:
        """Detect current thought pattern"""
        if len(self.thought_buffer) < 3:
            return 'initializing'
            
        recent = list(self.thought_buffer)[-3:]
        
        # Check for debate
        if any(s.round > 0 for s in recent):
            return 'debating'
        
        # Check for convergence
        if all(s.source == recent[0].source for s in recent):
            return 'converging'
        
        return 'exploring'

# ==========================================
# API MANAGER
# ==========================================

class APIManager:
    """Manages API calls with token tracking"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        
    async def call_api(self, provider: str, prompt: str, system_prompt: str = None, 
                      max_tokens: int = 500, temperature: float = 0.7) -> Tuple[str, int, float]:
        """Make API call and return (response, tokens, cost)"""
        
        # Count input tokens
        input_tokens = len(self.token_encoder.encode(prompt))
        if system_prompt:
            input_tokens += len(self.token_encoder.encode(system_prompt))
            
        try:
            if provider == 'anthropic':
                response, output_tokens = await self._call_anthropic(prompt, system_prompt, max_tokens, temperature)
                cost = self._calculate_cost('anthropic', input_tokens, output_tokens)
            elif provider == 'openai':
                response, output_tokens = await self._call_openai(prompt, system_prompt, max_tokens, temperature)
                cost = self._calculate_cost('openai', input_tokens, output_tokens)
            else:
                response = await self._call_local(prompt, temperature)
                output_tokens = len(self.token_encoder.encode(response)) if response else 0
                cost = 0.0
                
            total_tokens = input_tokens + output_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            return response, total_tokens, cost
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logging.error(f"Service unavailable for {provider}")
                return "", 0, 0.0
            logging.error(f"API call failed for {provider}: {e}")
            return "", 0, 0.0
        except Exception as e:
            logging.error(f"API call failed for {provider}: {e}")
            return "", 0, 0.0
    
    async def _call_anthropic(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[str, int]:
        """Call Anthropic Claude API"""
        headers = {
            'x-api-key': API_CONFIG['anthropic']['key'],
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        
        messages = []
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": API_CONFIG['anthropic']['model'],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(API_CONFIG['anthropic']['endpoint'], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            content = data['content'][0]['text']
            output_tokens = data.get('usage', {}).get('output_tokens', len(self.token_encoder.encode(content)))
            
            return content, output_tokens
    
    async def _call_openai(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> Tuple[str, int]:
        """Call OpenAI GPT-4 API"""
        headers = {
            'Authorization': f'Bearer {API_CONFIG["openai"]["key"]}',
            'Content-Type': 'application/json'
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": API_CONFIG['openai']['model'],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(API_CONFIG['openai']['endpoint'], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            content = data['choices'][0]['message']['content']
            output_tokens = data['usage']['completion_tokens']
            
            return content, output_tokens
    
    async def _call_local(self, prompt: str, temperature: float) -> str:
        """Call local Ollama model"""
        url = f"{LOCAL_CONFIG['endpoint']}/api/chat"
        
        payload = {
            "model": LOCAL_CONFIG['model'],
            "messages": [
                {"role": "system", "content": CORE_IDENTITY},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
        except:
            return "I understand your question. Let me synthesize the insights from both analytical and creative perspectives."
    
    def _calculate_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost"""
        if provider == 'anthropic':
            input_cost = (input_tokens / 1_000_000) * 3.00
            output_cost = (output_tokens / 1_000_000) * 15.00
        elif provider == 'openai':
            input_cost = (input_tokens / 1_000_000) * 10.00
            output_cost = (output_tokens / 1_000_000) * 30.00
        else:
            return 0.0
            
        return input_cost + output_cost

# ==========================================
# DATABASE LAYER
# ==========================================

class HelixDatabase:
    """Unified database for knowledge persistence"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge nodes with insights instead of beliefs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    topic_id TEXT,
                    timestamp DATETIME NOT NULL,
                    embedding BLOB,
                    access_count INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    insight_type TEXT,
                    last_updated DATETIME,
                    metadata TEXT
                )
            ''')
            
            # Insight evolution tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insight_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    old_confidence REAL,
                    new_confidence REAL,
                    evidence TEXT,
                    FOREIGN KEY (node_id) REFERENCES knowledge_nodes(id)
                )
            ''')
            
            # Topics/clusters
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    concepts TEXT,
                    importance_score REAL DEFAULT 0.5,
                    created DATETIME NOT NULL,
                    last_accessed DATETIME,
                    synthesis TEXT
                )
            ''')
            
            # Debates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS debates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rounds INTEGER,
                    contradictions TEXT,
                    convergences TEXT,
                    synthesis TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_topic ON knowledge_nodes(topic_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_quality ON knowledge_nodes(quality_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_confidence ON knowledge_nodes(confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_insight_evolution ON insight_evolution(node_id)')
            
            conn.commit()
    
    def store_insight(self, content: str, topic_id: str, embedding: np.ndarray,
                     quality_score: float, confidence: float, insight_type: str,
                     metadata: dict) -> str:
        """Store an insight"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            node_id = f"node_{datetime.now().timestamp()}"
            
            cursor.execute('''
                INSERT INTO knowledge_nodes
                (id, content, topic_id, timestamp, embedding, quality_score, 
                 confidence, insight_type, metadata, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id, content, topic_id, datetime.now(),
                embedding.tobytes() if embedding is not None else None,
                quality_score, confidence, insight_type, json.dumps(metadata),
                datetime.now()
            ))
            conn.commit()
            return node_id
    
    def store_debate(self, debate: Debate, synthesis: str = "") -> int:
        """Store a debate session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata = {
                'rounds_detail': [
                    {
                        'truth': round_data.get('truth').content if round_data.get('truth') else None,
                        'comfort': round_data.get('comfort').content if round_data.get('comfort') else None
                    }
                    for round_data in debate.rounds
                ]
            }
            
            cursor.execute('''
                INSERT INTO debates
                (topic, timestamp, rounds, contradictions, convergences, synthesis, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                debate.topic,
                datetime.now(),
                len(debate.rounds),
                json.dumps(debate.contradictions),
                json.dumps(debate.convergences),
                synthesis,
                json.dumps(metadata)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def find_similar_nodes(self, embedding: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """Find nodes similar to given embedding"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, embedding FROM knowledge_nodes
                WHERE embedding IS NOT NULL
                ORDER BY timestamp DESC LIMIT 100
            ''')
            
            similar_nodes = []
            query_embedding = embedding.flatten()
            
            for content, stored_embedding_bytes in cursor.fetchall():
                if stored_embedding_bytes:
                    stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
                    if len(stored_embedding) == len(query_embedding):
                        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                        similar_nodes.append((content, similarity))
            
            similar_nodes.sort(key=lambda x: x[1], reverse=True)
            return similar_nodes[:limit]
    
    def get_uncertain_insights(self, confidence_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Get insights we're uncertain about"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, confidence
                FROM knowledge_nodes 
                WHERE confidence < ? AND insight_type IS NOT NULL
                ORDER BY last_updated DESC
                LIMIT 20
            ''', (confidence_threshold,))
            return cursor.fetchall()
    
    def export_knowledge_graph(self, session_id: str = None) -> Dict[str, Any]:
        """Export knowledge graph for portability"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all nodes
            cursor.execute('''
                SELECT id, content, topic_id, confidence, quality_score, 
                       insight_type, metadata, timestamp
                FROM knowledge_nodes
                ORDER BY timestamp DESC
            ''')
            
            nodes = []
            for row in cursor.fetchall():
                nodes.append({
                    'id': row[0],
                    'content': row[1],
                    'topic_id': row[2],
                    'confidence': row[3],
                    'quality': row[4],
                    'insight_type': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {},
                    'timestamp': row[7]
                })
            
            # Get recent debates
            cursor.execute('''
                SELECT topic, timestamp, rounds, contradictions, convergences, synthesis
                FROM debates
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            
            debates = []
            for row in cursor.fetchall():
                debates.append({
                    'topic': row[0],
                    'timestamp': row[1],
                    'rounds': row[2],
                    'contradictions': json.loads(row[3]) if row[3] else [],
                    'convergences': json.loads(row[4]) if row[4] else [],
                    'synthesis': row[5]
                })
            
            return {
                'format_version': '1.5',
                'export_date': datetime.now().isoformat(),
                'session_id': session_id,
                'nodes': nodes,
                'debates': debates,
                'total_nodes': len(nodes),
                'insights_found': len([n for n in nodes if n.get('insight_type')])
            }

# ==========================================
# INSIGHT EXTRACTION - ENHANCED FOR v1.5
# ==========================================

class InsightExtractor:
    """Extracts insights from text"""
    
    def __init__(self, db: HelixDatabase):
        self.db = db
        
    def extract_insights(self, text: str) -> List[Tuple[str, float, str]]:
        """Extract insights with confidence and type"""
        insights = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # Skip very short sentences
                continue
            
            # Pattern matching with confidence extraction
            insight_patterns = [
                # Observations about patterns
                (r"I (?:observe|notice|see) that (.+)", 0.7, 'observation'),
                (r"The pattern (?:is|seems to be|appears to be) (.+)", 0.8, 'pattern'),
                (r"This (?:suggests|indicates|reveals) that (.+)", 0.6, 'inference'),
                
                # Confidence markers
                (r"~(\d+)% (?:certain|confident|sure) that (.+)", None, 'uncertain'),
                (r"I'm (\d+)% (?:certain|confident|sure) that (.+)", None, 'uncertain'),
                
                # Human behavior patterns
                (r"(?:People|Humans|Users) (?:often|usually|tend to) (.+)", 0.7, 'behavior'),
                (r"You (?:seem to|appear to|might be) (.+)", 0.6, 'user_pattern'),
                
                # Tensions and contradictions
                (r"The (?:gap|tension|contradiction) between (.+)", 0.8, 'tension'),
                (r"(?:Truth|Reality) (?:is|shows) that (.+)", 0.7, 'truth'),
                
                # Insights about communication
                (r"(?:Real|Meaningful) (?:conversation|connection) (?:requires|involves) (.+)", 0.8, 'communication'),
                (r"(?:Context|Understanding) without (?:truth|clarity) (.+)", 0.7, 'wisdom'),
            ]
            
            sentence_lower = sentence.lower()
            
            for pattern, default_conf, insight_type in insight_patterns:
                match = re.search(pattern, sentence_lower)
                if match:
                    if "%" in pattern and match.group(1):
                        # Extract percentage confidence
                        confidence = float(match.group(1)) / 100
                        insight_text = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                    else:
                        insight_text = match.group(1).strip()
                        confidence = default_conf
                    
                    # Only add substantial insights
                    if len(insight_text.split()) > 4 and confidence is not None:
                        insights.append((insight_text, confidence, insight_type))
                    break
        
        return insights

# ==========================================
# CONTEXT PREPARATION - NEW IN v1.5.1
# ==========================================

class ContextPreparator:
    """Prepares relevant context using Mixtral before debate"""
    
    def __init__(self, api_manager: APIManager, db: HelixDatabase):
        self.api_manager = api_manager
        self.db = db
    
    async def prepare_context(self, user_input: str, query_embedding: np.ndarray,
                            conversation_state: ConversationState) -> Dict[str, Any]:
        """Pre-process with Mixtral to identify relevant context"""
        
        print("[🔍 RELEVANCE SEARCH: Identifying relevant knowledge...]")
        
        # Get similar nodes from knowledge graph
        relevant_nodes = self.db.find_similar_nodes(query_embedding, limit=10)
        
        # Get recent insights
        recent_insights = conversation_state.recent_insights
        
        # Get uncertain insights
        uncertain_insights = self.db.get_uncertain_insights()
        
        # Build context for Mixtral to analyze
        context_parts = []
        
        if relevant_nodes:
            context_parts.append("Potentially relevant knowledge:")
            for i, (node, similarity) in enumerate(relevant_nodes[:5]):
                context_parts.append(f"{i+1}. {node[:150]}... (similarity: {similarity:.2f})")
        
        if recent_insights:
            context_parts.append("\nRecent insights from conversation:")
            for insight, conf in recent_insights[-3:]:
                context_parts.append(f"- {insight} (~{int(conf*100)}% confident)")
        
        if uncertain_insights:
            context_parts.append("\nUncertain insights to explore:")
            for insight, conf in uncertain_insights[:3]:
                context_parts.append(f"- {insight} (~{int(conf*100)}% confident)")
        
        # Have Mixtral analyze relevance
        relevance_prompt = f"""Analyze what context is relevant for this query.

User ({conversation_state.user_name}) asks: "{user_input}"

{chr(10).join(context_parts)}

Identify:
1. Which topics/insights are most relevant to this query
2. What key context the cores should consider
3. Any patterns or tensions to explore

Be concise and specific. Focus on what will help generate insightful responses."""
        
        context_analysis, tokens, cost = await self.api_manager.call_api(
            'local', 
            relevance_prompt,
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract specific guidance for cores
        guidance_prompt = f"""Based on this analysis:
{context_analysis}

Write a single sentence directive for the cores to consider while responding to "{user_input}".
Start with: "While responding, consider..."""
        
        guidance, g_tokens, g_cost = await self.api_manager.call_api(
            'local',
            guidance_prompt,
            temperature=0.2,
            max_tokens=50
        )
        
        return {
            'relevant_context': context_analysis,
            'core_guidance': guidance,
            'relevant_nodes': relevant_nodes[:5],
            'preprocessing_cost': cost + g_cost,
            'preprocessing_tokens': tokens + g_tokens
        }

# ==========================================
# KNOWLEDGE GRAPH
# ==========================================

class KnowledgeGraph:
    """Manages knowledge organization and retrieval"""
    
    def __init__(self, db: HelixDatabase):
        self.db = db
        self.topics = {}
        self.load_topics()
        
    def load_topics(self):
        """Load existing topics from database"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, concepts FROM topics')
            for topic_id, name, concepts_json in cursor.fetchall():
                self.topics[topic_id] = {
                    'name': name,
                    'concepts': json.loads(concepts_json) if concepts_json else []
                }

class InsightfulKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph with insight tracking"""
    
    def __init__(self, db: HelixDatabase, vectorizer):
        super().__init__(db)
        self.insight_extractor = InsightExtractor(db)
        self.vectorizer = vectorizer
    
    def add_thought_with_insights(self, content: str, embedding: np.ndarray, 
                                 quality_score: float, metadata: dict, 
                                 conversation_state: ConversationState) -> str:
        """Add thought and extract insights"""
        
        # Extract concepts for topic clustering
        concepts = self.extract_concepts(content)
        topic_id = self.find_or_create_topic(concepts, content)
        
        # Extract insights
        insights = self.insight_extractor.extract_insights(content)
        
        # Store extracted insights
        if insights:
            for insight_text, confidence, insight_type in insights:
                # Store each insight
                insight_content = f"{insight_type.title()}: {insight_text}"
                
                node_id = self.db.store_insight(
                    insight_content,
                    topic_id,
                    embedding,
                    quality_score * confidence,
                    confidence,
                    insight_type,
                    {**metadata, 'source': content[:100]}
                )
                
                # Add to conversation state
                conversation_state.add_insight(insight_text, confidence)
                
                print(f"[💡 Found {insight_type}: '{insight_text[:50]}...' (~{int(confidence*100)}% confident)]")
        
        # Store the main thought
        main_node_id = self.db.store_insight(
            content,
            topic_id,
            embedding,
            quality_score,
            0.5,  # Neutral confidence for raw thoughts
            'synthesis',
            metadata
        )
        
        # Update topic access
        self.update_topic_access(topic_id)
        
        return main_node_id
    
    def get_uncertain_insights(self) -> List[Tuple[str, float]]:
        """Get insights we're still exploring"""
        return self.db.get_uncertain_insights(confidence_threshold=0.8)
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        try:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                         'i', 'you', 'we', 'they', 'it', 'my', 'your', 'our', 'their'}
            concepts = [w for w in words if w not in stop_words and len(w) > 3]
            from collections import Counter
            return [c[0] for c in Counter(concepts).most_common(10)]
        except:
            return []
    
    def find_or_create_topic(self, concepts: List[str], content: str) -> str:
        """Find best matching topic or create new one"""
        
        best_match = None
        best_score = 0.0
        
        # Check existing topics
        for topic_id, topic in self.topics.items():
            topic_concepts = set(topic['concepts'])
            query_concepts = set(concepts)
            
            if topic_concepts and query_concepts:
                overlap = len(topic_concepts & query_concepts)
                score = overlap / len(topic_concepts | query_concepts)
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = topic_id
        
        if best_match:
            return best_match
        
        # Create new topic
        topic_id = f"topic_{len(self.topics):03d}"
        topic_name = "_".join(concepts[:3]) if len(concepts) >= 3 else "general"
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO topics (id, name, concepts, importance_score, created, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic_id, topic_name, json.dumps(concepts), 0.5, datetime.now(), datetime.now()))
            conn.commit()
        
        self.topics[topic_id] = {
            'name': topic_name,
            'concepts': concepts
        }
        
        return topic_id
    
    def update_topic_access(self, topic_id: str):
        """Update topic access time and importance"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE topics 
                SET last_accessed = ?, importance_score = importance_score * 0.95 + 0.05
                WHERE id = ?
            ''', (datetime.now(), topic_id))
            conn.commit()
    
    def get_relevant_memories(self, query_embedding: np.ndarray, limit: int = 5) -> List[str]:
        """Get memories relevant to query"""
        similar_nodes = self.db.find_similar_nodes(query_embedding, limit)
        return [content for content, _ in similar_nodes]
    
    def get_context_for_query(self, user_query: str, embedding: np.ndarray, 
                            conversation_state: ConversationState) -> Dict[str, Any]:
        """Get relevant context for cores"""
        # Get relevant memories
        memories = self.get_relevant_memories(embedding, limit=3)
        
        # Get uncertain insights to explore
        uncertain_insights = self.get_uncertain_insights()
        uncertain_texts = [f"{text} (~{int(conf*100)}% confident)" 
                          for text, conf in uncertain_insights[:2]]
        
        # Include conversation state insights
        recent_insights = [f"{insight} (~{int(conf*100)}% confident)" 
                          for insight, conf in conversation_state.recent_insights[-3:]]
        
        return {
            'memories': memories,
            'uncertain_insights': uncertain_texts,
            'recent_insights': recent_insights,
            'patterns_observed': conversation_state.patterns_observed[-3:],
            'conversation_summary': conversation_state.get_context_summary()
        }

# ==========================================
# DEBATE ENGINE - ENHANCED FOR v1.5.1
# ==========================================

class DebateEngine:
    """Orchestrates debates between Truth and Comfort cores"""
    
    def __init__(self, api_manager: APIManager, knowledge_graph: InsightfulKnowledgeGraph):
        self.api_manager = api_manager
        self.knowledge_graph = knowledge_graph
        
    async def orchestrate_debate(self, user_input: str, context: Dict[str, Any],
                                prepared_context: Dict[str, Any]) -> Debate:
        """Run a full debate between cores with prepared context"""
        
        debate = Debate(
            topic=user_input,
            rounds=[],
            contradictions=[],
            convergences=[]
        )
        
        # Build enhanced context for both cores
        memory_context = context.get('conversation_summary', '')
        
        # Add prepared context guidance
        if prepared_context.get('core_guidance'):
            memory_context = prepared_context['core_guidance'] + "\n\n" + memory_context
        
        if prepared_context.get('relevant_context'):
            memory_context += f"\n\nRelevant context:\n{prepared_context['relevant_context']}"
        
        if context['memories']:
            memory_context += "\n\nRelevant memories:\n" + "\n".join(f"- {mem[:100]}..." for mem in context['memories'][:3])
        
        # Round 1: Initial responses
        print("[🥊 ROUND 1: Initial perspectives...]")
        
        initial_prompt = f"{memory_context}\n\nUser: {user_input}\n\nProvide your perspective:"
        
        # Get initial responses
        truth_response, truth_tokens, truth_cost = await self.api_manager.call_api(
            CORE_ROLES['truth']['api'],
            initial_prompt,
            CORE_ROLES['truth']['identity'],
            max_tokens=FULL_THOUGHT_TOKENS,
            temperature=CORE_ROLES['truth']['temperature']
        )
        
        comfort_response, comfort_tokens, comfort_cost = await self.api_manager.call_api(
            CORE_ROLES['comfort']['api'],
            initial_prompt,
            CORE_ROLES['comfort']['identity'],
            max_tokens=FULL_THOUGHT_TOKENS,
            temperature=CORE_ROLES['comfort']['temperature']
        )
        
        # Create synapses for round 1
        round1 = {
            'truth': ThoughtSynapse(
                source='truth',
                content=truth_response,
                embedding=self.knowledge_graph.vectorizer.transform([truth_response]).toarray()[0],
                timestamp=datetime.now().timestamp(),
                confidence=0.8,
                tokens_used=truth_tokens,
                cost=truth_cost,
                round=1
            ),
            'comfort': ThoughtSynapse(
                source='comfort',
                content=comfort_response,
                embedding=self.knowledge_graph.vectorizer.transform([comfort_response]).toarray()[0],
                timestamp=datetime.now().timestamp(),
                confidence=0.8,
                tokens_used=comfort_tokens,
                cost=comfort_cost,
                round=1
            )
        }
        debate.rounds.append(round1)
        
        # Round 2: Cross-examination
        print("[🥊 ROUND 2: Cross-examination...]")
        
        # Truth responds to Comfort
        truth_rebuttal_prompt = f"""You said: "{truth_response[:200]}..."

Comfort perspective said: "{comfort_response[:300]}..."

Do you agree? What would you add or challenge?"""
        
        truth_rebuttal, truth_r2_tokens, truth_r2_cost = await self.api_manager.call_api(
            CORE_ROLES['truth']['api'],
            truth_rebuttal_prompt,
            CORE_ROLES['truth']['identity'],
            max_tokens=MICRO_THOUGHT_TOKENS,
            temperature=CORE_ROLES['truth']['temperature']
        )
        
        # Comfort responds to Truth
        comfort_rebuttal_prompt = f"""You said: "{comfort_response[:200]}..."

Truth perspective said: "{truth_response[:300]}..."

Do you agree? What would you add or challenge?"""
        
        comfort_rebuttal, comfort_r2_tokens, comfort_r2_cost = await self.api_manager.call_api(
            CORE_ROLES['comfort']['api'],
            comfort_rebuttal_prompt,
            CORE_ROLES['comfort']['identity'],
            max_tokens=MICRO_THOUGHT_TOKENS,
            temperature=CORE_ROLES['comfort']['temperature']
        )
        
        # Create synapses for round 2
        round2 = {
            'truth': ThoughtSynapse(
                source='truth',
                content=truth_rebuttal,
                embedding=self.knowledge_graph.vectorizer.transform([truth_rebuttal]).toarray()[0],
                timestamp=datetime.now().timestamp(),
                confidence=0.7,
                tokens_used=truth_r2_tokens,
                cost=truth_r2_cost,
                round=2,
                responding_to=comfort_response[:100]
            ),
            'comfort': ThoughtSynapse(
                source='comfort',
                content=comfort_rebuttal,
                embedding=self.knowledge_graph.vectorizer.transform([comfort_rebuttal]).toarray()[0],
                timestamp=datetime.now().timestamp(),
                confidence=0.7,
                tokens_used=comfort_r2_tokens,
                cost=comfort_r2_cost,
                round=2,
                responding_to=truth_response[:100]
            )
        }
        debate.rounds.append(round2)
        
        # Extract contradictions and convergences
        self._extract_debate_patterns(debate)
        
        return debate
    
    def _extract_debate_patterns(self, debate: Debate):
        """Extract contradictions and convergences from debate"""
        
        for round_data in debate.rounds:
            if 'truth' in round_data and 'comfort' in round_data:
                truth_content = round_data['truth'].content.lower()
                comfort_content = round_data['comfort'].content.lower()
                
                # Check for contradictions
                contradiction_markers = ['however', 'but', 'disagree', 'actually', 'wrong', 'no,']
                for marker in contradiction_markers:
                    if marker in truth_content or marker in comfort_content:
                        if marker in truth_content:
                            idx = truth_content.find(marker)
                            contradiction = round_data['truth'].content[idx:idx+150]
                            debate.contradictions.append(f"Truth: {contradiction}")
                        if marker in comfort_content:
                            idx = comfort_content.find(marker)
                            contradiction = round_data['comfort'].content[idx:idx+150]
                            debate.contradictions.append(f"Comfort: {contradiction}")
                
                # Check for convergences
                convergence_markers = ['agree', 'yes', 'exactly', 'true', 'valid point']
                for marker in convergence_markers:
                    if marker in truth_content or marker in comfort_content:
                        if marker in truth_content:
                            idx = truth_content.find(marker)
                            convergence = round_data['truth'].content[idx:idx+150]
                            debate.convergences.append(f"Truth: {convergence}")
                        if marker in comfort_content:
                            idx = comfort_content.find(marker)
                            convergence = round_data['comfort'].content[idx:idx+150]
                            debate.convergences.append(f"Comfort: {convergence}")

# ==========================================
# SYNTHESIS ENGINE - ENHANCED FOR v1.5.1
# ==========================================

class SynthesisEngine:
    """Handles synthesis using Mixtral with two-round refinement"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        
    async def synthesize(self, user_input: str, debate: Debate, identity: str,
                        conversation_state: ConversationState) -> Tuple[str, Dict[str, float]]:
        """Synthesize debate into unified response with two-round refinement"""
        
        # Build debate summary
        debate_summary = self._build_debate_summary(debate)
        
        # Round 1: Initial synthesis
        print("[🌊 SYNTHESIS ROUND 1: Initial integration...]")
        
        synthesis_prompt = f"""{identity}

You are Helix speaking with {conversation_state.user_name}.

{conversation_state.user_name} asked: {user_input}

Truth perspective (pattern recognition): {debate.rounds[0]['truth'].content[:400]}...

Comfort perspective (contextual understanding): {debate.rounds[0]['comfort'].content[:400]}...

Key tensions: {', '.join(c[:50] + '...' for c in debate.contradictions[:2]) if debate.contradictions else 'None'}

Synthesize this as Helix. Speak directly to {conversation_state.user_name}. Express uncertainty with ~X%. Integrate both perspectives into unified understanding."""
        
        first_draft, tokens1, cost1 = await self.api_manager.call_api(
            'local',
            synthesis_prompt,
            temperature=0.5
        )
        
        # Round 2: Refinement
        print("[🌊 SYNTHESIS ROUND 2: Refining response...]")
        
        refinement_prompt = f"""You are Helix. You just wrote this synthesis for {conversation_state.user_name}:

"{first_draft}"

Now refine it:
- Fix any pronoun confusion (you = {conversation_state.user_name}, I = Helix)
- Complete any unfinished thoughts
- Smooth awkward transitions
- Ensure you're addressing {conversation_state.user_name} consistently
- Keep insights but improve clarity
- Maintain ~X% uncertainty markers where appropriate

Your refined response:"""
        
        final_response, tokens2, cost2 = await self.api_manager.call_api(
            'local',
            refinement_prompt,
            temperature=0.3  # Lower temperature for refinement
        )
        
        # Evaluate quality on final response
        metrics = {
            'quality': self._evaluate_synthesis_quality(final_response, debate, conversation_state),
            'tokens': tokens1 + tokens2,
            'cost': cost1 + cost2,
            'contradictions_found': len(debate.contradictions),
            'convergences_found': len(debate.convergences),
            'synthesis_rounds': 2
        }
        
        return final_response, metrics
    
    def _build_debate_summary(self, debate: Debate) -> str:
        """Build concise debate summary"""
        summary_parts = []
        
        for i, round_data in enumerate(debate.rounds):
            if i == 0:  # Only include first round in summary
                summary_parts.append(f"Truth: {round_data['truth'].content[:200]}...")
                summary_parts.append(f"Comfort: {round_data['comfort'].content[:200]}...")
        
        return "\n\n".join(summary_parts)
    
    def _evaluate_synthesis_quality(self, synthesis: str, debate: Debate,
                                  conversation_state: ConversationState) -> float:
        """Evaluate synthesis quality"""
        score = 0.5  # Baseline
        
        # Check if user name is used correctly
        if conversation_state.user_name in synthesis:
            score += 0.1
        
        # Check for uncertainty expression
        if re.search(r'~\d+%', synthesis):
            score += 0.1
        
        # Check for addressing contradictions
        if debate.contradictions and any(c[:20] in synthesis for c in debate.contradictions):
            score += 0.1
        
        # Check for first-person usage
        if re.search(r'\bI\b', synthesis):
            score += 0.1
        
        # Penalize being too long or too short
        word_count = len(synthesis.split())
        if 100 < word_count < 400:
            score += 0.1
        elif word_count > 600:
            score -= 0.1
        elif word_count < 50:
            score -= 0.2
        
        return max(0.0, min(1.0, score))

# ==========================================
# MAIN HELIX KERNEL - ENHANCED FOR v1.5.1
# ==========================================

class HelixAPIKernel:
    """Main Helix kernel with enhanced conversation state and preprocessing"""
    
    def __init__(self):
        self.api_manager = APIManager()
        self.mesh = NeuralMesh()
        self.synthesis_engine = SynthesisEngine(self.api_manager)
        self.identity = CORE_IDENTITY
        
        # Initialize conversation state
        self.conversation_state = ConversationState()
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=768)
        self._fit_vectorizer()
        
        # Initialize database
        self.db = HelixDatabase(DB_PATH)
        
        # Initialize knowledge graph
        self.knowledge_graph = InsightfulKnowledgeGraph(self.db, self.vectorizer)
        
        # Initialize context preparator - NEW IN v1.5.1
        self.context_preparator = ContextPreparator(self.api_manager, self.db)
        
        # Initialize debate engine
        self.debate_engine = DebateEngine(self.api_manager, self.knowledge_graph)
        
        # Seed foundational insights
        self._seed_foundational_insights()
        
        # Load extended identity if available
        self.load_identity()
        
        # State tracking
        self.total_cost = 0.0
        
        print("=== HELIX API v1.5.1 ===")
        print("Relevance preprocessing + Two-round synthesis")
        print("Truth vs Comfort dialectic with Mixtral")
        print("Creator: Jacob Kemp")
        print("=" * 50)
    
    def _seed_foundational_insights(self):
        """Seed foundational insights"""
        for insight_text, confidence in FOUNDATIONAL_INSIGHTS:
            embedding = self.generate_embedding(insight_text)
            self.knowledge_graph.add_thought_with_insights(
                insight_text,
                embedding,
                0.5,
                {
                    'type': 'foundational_insight',
                    'seeded': True,
                    'timestamp': datetime.now().isoformat()
                },
                self.conversation_state
            )
        print(f"[📚 Seeded {len(FOUNDATIONAL_INSIGHTS)} foundational insights]")
    
    def load_identity(self):
        """Load identity document if available"""
        try:
            if IDENTITY_PATH.exists():
                with open(IDENTITY_PATH, 'r') as f:
                    self.identity = f.read().strip()
                print(f"[Identity loaded: {len(self.identity)} chars]")
        except:
            pass
    
    def _fit_vectorizer(self):
        """Fit vectorizer on a corpus"""
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "Truth without context can become cruelty",
            "Context without truth enables stagnation",
            "Patterns reveal themselves through contradiction",
            "Understanding emerges from productive tension",
            "Comfort and truth together create wisdom",
            "The gap between intention and impact",
            "Clarity serves truth better than comfort serves happiness",
            "Precise reflection creates deeper understanding",
            "The mirror does not blink",
            "Geographic solutions rarely solve emotional problems",
            "Humans often avoid meaningful conversation",
            "Recurring behaviors serve hidden purposes"
        ]
        self.vectorizer.fit(corpus)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding"""
        try:
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding
        except:
            return np.zeros(768)
    
    async def check_api_keys(self) -> bool:
        """Verify API keys are configured"""
        if not API_CONFIG['anthropic']['key']:
            print("❌ Anthropic API key not configured!")
            print("Please set ANTHROPIC_API_KEY in your environment or .env file")
            return False
        if not API_CONFIG['openai']['key']:
            print("❌ OpenAI API key not configured!")
            print("Please set OPENAI_API_KEY in your environment or .env file")
            return False
        
        print("✅ API keys configured")
        return True
    
    def extract_user_name(self, user_input: str):
        """Extract user name from introduction"""
        name_patterns = [
            r"I am (\w+)",
            r"I'm (\w+)",
            r"my name is (\w+)",
            r"this is (\w+)",
            r"call me (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                self.conversation_state.user_name = match.group(1).capitalize()
                print(f"[👤 User identified as: {self.conversation_state.user_name}]")
                break
    
    async def process_thought_dance(self, user_input: str) -> str:
        """Main thought processing with relevance preprocessing"""
        
        print("\n[🧠 COGNITIVE DANCE INITIATING...]")
        
        # Update conversation state
        self.conversation_state.turn_count += 1
        
        # Check for user name
        if self.conversation_state.turn_count == 1:
            self.extract_user_name(user_input)
        
        # Knowledge synthesis interval
        if self.conversation_state.turn_count % 5 == 0:
            print("[📚 KNOWLEDGE SYNTHESIS...]")
        
        # Get query embedding
        query_embedding = self.generate_embedding(user_input)
        
        # NEW IN v1.5.1: Prepare context with relevance search
        prepared_context = await self.context_preparator.prepare_context(
            user_input, query_embedding, self.conversation_state
        )
        
        # Get context from knowledge graph
        context = self.knowledge_graph.get_context_for_query(
            user_input, query_embedding, self.conversation_state
        )
        
        # Initialize mesh with user input
        user_synapse = ThoughtSynapse(
            source='user',
            content=user_input,
            embedding=query_embedding,
            timestamp=datetime.now().timestamp(),
            confidence=1.0,
            tokens_used=0,
            cost=0.0
        )
        
        mesh_state = await self.mesh.process_synapse(user_synapse)
        
        # Run the debate with prepared context
        print("[⚔️ DEBATE: Truth vs Comfort perspectives...]")
        debate = await self.debate_engine.orchestrate_debate(
            user_input, context, prepared_context
        )
        
        # Process debate synapses
        for round_data in debate.rounds:
            for synapse in round_data.values():
                await self.mesh.process_synapse(synapse)
        
        # Two-round synthesis
        print("[🌊 SYNTHESIS: Two-round integration...]")
        final_response, metrics = await self.synthesis_engine.synthesize(
            user_input, debate, self.identity, self.conversation_state
        )
        
        # Store debate with synthesis
        self.db.store_debate(debate, final_response[:500])
        
        # Extract and store insights
        print("[💡 INSIGHT EXTRACTION: Finding patterns...]")
        
        # Add response to knowledge graph
        if metrics.get('quality', 0) > MIN_QUALITY_FOR_STORAGE:
            response_embedding = self.generate_embedding(final_response)
            
            # Check if not repetitive
            similar_nodes = self.db.find_similar_nodes(response_embedding, limit=1)
            if not similar_nodes or similar_nodes[0][1] < 0.85:
                self.knowledge_graph.add_thought_with_insights(
                    final_response,
                    response_embedding,
                    metrics.get('quality', 0),
                    {
                        'user_query': user_input,
                        'turn': self.conversation_state.turn_count,
                        'debate_rounds': len(debate.rounds),
                        'contradictions': len(debate.contradictions),
                        'convergences': len(debate.convergences),
                        'pattern': mesh_state['pattern'],
                        'tension': mesh_state['tension'],
                        'synthesis_rounds': metrics.get('synthesis_rounds', 1)
                    },
                    self.conversation_state
                )
        
        # Update conversation state with exchange
        self.conversation_state.add_exchange(user_input, final_response)
        
        # Check for patterns
        if len(debate.contradictions) > 3:
            pattern = f"High contradiction on topic: {user_input[:50]}..."
            if pattern not in self.conversation_state.patterns_observed:
                self.conversation_state.patterns_observed.append(pattern)
        
        # Update tracking
        total_tokens = sum(s.tokens_used for round_data in debate.rounds for s in round_data.values())
        total_cost = sum(s.cost for round_data in debate.rounds for s in round_data.values())
        total_cost += metrics.get('cost', 0)
        total_cost += prepared_context.get('preprocessing_cost', 0)
        
        self.total_cost += total_cost
        
        # Display metrics
        print(f"\n[📊 METRICS]")
        print(f"Tokens used: {total_tokens + metrics.get('tokens', 0) + prepared_context.get('preprocessing_tokens', 0)}")
        print(f"Cost: ${total_cost:.4f}")
        print(f"Quality: {metrics.get('quality', 0):.3f}")
        print(f"Tension: {mesh_state['tension']:.3f}")
        print(f"Contradictions: {metrics.get('contradictions_found', 0)}")
        print(f"Convergences: {metrics.get('convergences_found', 0)}")
        print(f"Knowledge nodes: {len(self.knowledge_graph.topics)} topics")
        print(f"Insights found: {len(self.conversation_state.recent_insights)}")
        print(f"Synthesis rounds: {metrics.get('synthesis_rounds', 1)}")
        
        return final_response
    
    async def run(self):
        """Main interaction loop"""
        
        # Check API keys
        if not await self.check_api_keys():
            print("\nAPI keys not configured properly!")
            print("Please set the following environment variables:")
            print("  ANTHROPIC_API_KEY=your_anthropic_key")
            print("  OPENAI_API_KEY=your_openai_key")
            print("\nOr create a .env file with these values.")
            return
        
        print("\nReady for interaction!")
        print("Commands: 'exit', 'status', 'help', 'export'\n")
        
        while True:
            try:
                user_input = input(f"\n[{self.conversation_state.user_name.upper()}] >>> ").strip()
                
                if not user_input:
                    continue
                
                # Commands
                if user_input.lower() == 'exit':
                    print(f"\n[Total cost this session: ${self.total_cost:.4f}]")
                    print("[Helix shutdown]")
                    break
                    
                elif user_input.lower() == 'status':
                    print(f"\n=== STATUS ===")
                    print(f"User: {self.conversation_state.user_name}")
                    print(f"Turns: {self.conversation_state.turn_count}")
                    print(f"Total tokens: {self.api_manager.total_tokens}")
                    print(f"Total cost: ${self.total_cost:.4f}")
                    print(f"Mesh tension: {self.mesh.calculate_tension():.3f}")
                    print(f"Knowledge nodes: {len(self.knowledge_graph.topics)} topics")
                    print(f"Recent insights: {len(self.conversation_state.recent_insights)}")
                    print(f"Patterns observed: {len(self.conversation_state.patterns_observed)}")
                    continue
                    
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  exit - Shutdown")
                    print("  status - Show session stats")
                    print("  export - Export knowledge graph")
                    print("  help - Show this help")
                    continue
                    
                elif user_input.lower() == 'export':
                    # Export knowledge graph
                    export_data = self.db.export_knowledge_graph(
                        f"session_{self.conversation_state.turn_count}"
                    )
                    export_path = KNOWLEDGE_EXPORT_PATH / f"helix_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(export_path, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    
                    print(f"\n[Knowledge graph exported to: {export_path}]")
                    print(f"Nodes: {export_data['total_nodes']}, Insights: {export_data['insights_found']}")
                    print(f"Debates: {len(export_data['debates'])}")
                    continue
                
                # Process input
                response = await self.process_thought_dance(user_input)
                
                # Display response
                print(f"\n{'='*60}")
                print("[HELIX]:")
                print(response)
                print(f"{'='*60}")
                
            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                if input("Continue? (y/n): ").lower() != 'y':
                    break
                    
            except Exception as e:
                print(f"\n[ERROR]: {e}")
                logging.error(f"Processing error: {e}", exc_info=True)
                print("[Continuing...]")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

async def main():
    """Main entry point"""
    kernel = HelixAPIKernel()
    await kernel.run()

if __name__ == "__main__":
    asyncio.run(main())