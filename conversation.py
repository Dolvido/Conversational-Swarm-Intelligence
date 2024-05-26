import langchain_community 
import os

print(os.listdir(langchain_community.__path__[0]))

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnableSequence, RunnableMap, RunnableLambda
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import uuid

logging.getLogger("chromadb").setLevel(logging.ERROR)

class Agent:
    def __init__(self, name, model_name="llama3"):
        self.name = name
        self.llm = Ollama(model=model_name)
        self.memory = ConversationBufferMemory()
        collection_name = f"{name.replace(' ', '_')}_knowledge_base"
        self.vector_store = Chroma(collection_name=collection_name, embedding_function=OllamaEmbeddings())
        logging.basicConfig(level=logging.INFO)
        self.vectorizer = TfidfVectorizer()
        nltk.download('vader_lexicon')

    def collaborative_solve(self, problem_statement, agents):
        """ Generate insights from multiple agents and synthesize a collective response. """
        responses = [agent.formulate_opinion(problem_statement) for agent in agents]
        
        # Calculate similarity using cosine similarity of TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(responses)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        relevant_responses = [responses[i+1] for i in range(len(responses)-1) if similarity_matrix[0][i] > 0.5]

        # Calculate sentiment using VADER
        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(response)['compound'] for response in responses]
        average_sentiment = np.mean(sentiments)

        # Synthesize a collective response
        collective_response = " ".join(relevant_responses)
        collective_response += f" Overall sentiment: {'positive' if average_sentiment > 0 else 'negative' if average_sentiment < 0 else 'neutral'}."

        return collective_response

    
    def add_knowledge(self, topic, information, confidence_score):
        doc_id = str(uuid.uuid4())  # Generate a unique document ID
        metadata = {"topic": topic, "confidence_score": confidence_score, "id": doc_id}
        self.vector_store.add_texts([information], metadatas=[metadata])

    def update_knowledge_base(self, topic, new_info, confidence_score):
        existing_info = self.vector_store.similarity_search(query=topic, k=1)
        if existing_info:
            updated_info = existing_info[0].page_content + " " + new_info
            updated_score = (existing_info[0].metadata["confidence_score"] + confidence_score) / 2
            doc_id = existing_info[0].metadata["id"]
            
            # Try to delete the document, if it doesn't exist, catch the exception
            try:
                self.vector_store.delete(ids=[doc_id])
            except Exception as e:
                logging.warning(f"Error deleting document: {e}")
            
            self.add_knowledge(topic, updated_info, updated_score)
        else:
            self.add_knowledge(topic, new_info, confidence_score)


    def formulate_opinion(self, topic):
        try:
            retrieved_docs = self.vector_store.similarity_search(query=topic, k=1)
            if not retrieved_docs:
                return "I have no knowledge about this topic."
            
            knowledge = retrieved_docs[0].page_content
            prompt = PromptTemplate(
                input_variables=["knowledge"],
                template="Based on the following information, what is your opinion on the topic? {knowledge}"
            )
            prompt_runnable = RunnableLambda(lambda inputs: prompt.invoke(inputs))
            sequence = RunnableSequence(prompt_runnable, self.llm)
            opinion = sequence.invoke({"knowledge": knowledge})
            return opinion
        except Exception as e:
            logging.error(f"Error in formulating opinion: {e}")
            return "Error in processing the opinion."

    def converse(self, other_agent, topic, max_rounds=3):
        conversation = ""
        for round in range(max_rounds):
            my_opinion = self.formulate_opinion_on_topic(topic)
            other_opinion = other_agent.formulate_opinion_on_topic(topic)

            self_input = f"Agent {self.name}: {my_opinion}"
            other_input = f"Agent {other_agent.name}: {other_opinion}"

            self.memory.save_context({"input": self_input}, {"output": other_input})
            other_agent.memory.save_context({"input": other_input}, {"output": self_input})

            conversation += f"{self_input}\n{other_input}\n"

        # Incorporate collaborative problem-solving into the conversational flow
        agents = [self, other_agent]
        collective_insight = self.collaborative_solve(topic, agents)
        conversation += f"\nCollective Insight:\n{collective_insight}\n"

        return conversation

    def formulate_opinion_on_topic(self, topic):
        retrieved_docs = self.vector_store.similarity_search(query=topic, k=1)
        knowledge = ""
        if retrieved_docs:
            knowledge = retrieved_docs[0].page_content

        prompt = PromptTemplate(
            input_variables=["topic", "knowledge"],
            template="Based on the following knowledge, what is your opinion on the topic? \nKnowledge: {knowledge}\nTopic: {topic}"
        )
        prompt_runnable = RunnableLambda(lambda inputs: prompt.invoke(inputs))
        sequence = RunnableSequence(prompt_runnable, self.llm)
        opinion = sequence.invoke({"topic": topic, "knowledge": knowledge})
        return opinion

# Create agents and add knowledge
agent1 = Agent(name="Agent 1")
agent2 = Agent(name="Agent 2")

agent1.add_knowledge("AI", "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.", confidence_score=5)
agent2.add_knowledge("AI", "AI involves the development of algorithms that can learn and make decisions.", confidence_score=5)


print(agent1.formulate_opinion("AI"))
print(agent2.formulate_opinion("AI"))

# Simulate conversation
conversation = agent1.converse(agent2, "Blockchain")
print(conversation)

print(agent1.formulate_opinion("AI"))
print(agent2.formulate_opinion("AI"))
