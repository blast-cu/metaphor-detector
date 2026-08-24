from sqlalchemy import String, ForeignKey, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import List

"""
sqlalchemy setup for the databases created in 'src.data_loading'
"""

class Base(DeclarativeBase):
    pass

class Document(Base): 
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # docIDX
    text: Mapped[str] = mapped_column(String)
    clean_text: Mapped[str] = mapped_column(String)
    num_sentences: Mapped[int] = mapped_column(Integer)


    sentences: Mapped[List["Sentence"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    coreference_chains: Mapped[List["CoreferenceChain"]] = relationship(    
        back_populates="document", cascade="all, delete-orphan"
    )
    doc_annotations: Mapped[List["DocAnnotation"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Document(id: {self.id}, text: {self.text[:30]}...)"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "clean_text": self.clean_text,
            "coreference_chains": [chain.to_dict() for chain in self.coreference_chains],
            "sentences": [sentence.to_dict() for sentence in self.sentences]
        }

class Sentence(Base):
    __tablename__ = "sentences"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # docIDX_sentIDX
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    text: Mapped[str] = mapped_column(String)
    clean_text: Mapped[str] = mapped_column(String)
    num_tokens: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)

    tokens: Mapped[List["Token"]] = relationship(
        back_populates="sentence", cascade="all, delete-orphan"
    )
    document: Mapped["Document"] = relationship(back_populates="sentences")
    coref_mentions: Mapped[List["CoreferenceMention"]] = relationship(
        back_populates="sentence", cascade="all, delete-orphan"
    )

    sent_annotations: Mapped[List["SentAnnotation"]] = relationship(
        back_populates="sentence", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Sentence(id: {self.id}, text: {self.text[:30]}...)"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "clean_text": self.clean_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "coref_mentions": [mention.to_dict() for mention in self.coref_mentions]
        }


class Token(Base): 
    __tablename__ = "tokens"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # docID_sentIDX_tokenIDX
    sentence_id: Mapped[str] = mapped_column(ForeignKey("sentences.id"))
    text: Mapped[str] = mapped_column(String)
    pos: Mapped[str] = mapped_column(String)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)

    sentence: Mapped["Sentence"] = relationship(back_populates="tokens")

    def __repr__(self) -> str:
        return f"Token(id: '{self.id}', text: '{self.text}', pos: {self.pos}, start_char: {self.start_char}, end_char: {self.end_char})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sentence_id": self.sentence_id,
            "text": self.text,
            "pos": self.pos
        }
    
class CoreferenceChain(Base):
    __tablename__ = "coreference_chains"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # docIDX_corefChainIDX
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    representative_text: Mapped[str] = mapped_column(String)

    document: Mapped["Document"] = relationship(back_populates="coreference_chains")
    mentions: Mapped[List["CoreferenceMention"]] = relationship(
        back_populates="coreference_chain", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"CoreferenceChain(id: {self.id}, representative_text: {self.representative_text})"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "representative_text": self.representative_text,
            "mentions": [mention.to_dict() for mention in self.mentions]
        }
    
class CoreferenceMention(Base):
    __tablename__ = "coreference_mentions"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # docIDX_corefChainIDX_mentionIDX
    coreference_chain_id: Mapped[str] = mapped_column(ForeignKey("coreference_chains.id"))
    sentence_id: Mapped[str] = mapped_column(ForeignKey("sentences.id"))
    start_token_id: Mapped[str] = mapped_column(ForeignKey("tokens.id"))
    end_token_id: Mapped[str] = mapped_column(ForeignKey("tokens.id"))
    text: Mapped[str] = mapped_column(String)

    coreference_chain: Mapped["CoreferenceChain"] = relationship(back_populates="mentions")
    sentence: Mapped["Sentence"] = relationship()
    start_token: Mapped["Token"] = relationship(
        foreign_keys=[start_token_id]
    )
    end_token: Mapped["Token"] = relationship(
        foreign_keys=[end_token_id]
    )
    def __repr__(self) -> str:
        return f"CoreferenceMention(id: {self.id}, text: {self.text[:30]}...)"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "coreference_chain_id": self.coreference_chain_id,
            "sentence_id": self.sentence_id,
            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "text": self.text
        }

class DocAnnotation(Base):
    __tablename__ = "doc_annotations"
    id: Mapped[str] = mapped_column(String, primary_key=True) # docIDX_annotationType
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    annotation_type: Mapped[str] = mapped_column(String)
    annotation_value: Mapped[str] = mapped_column(String)

    document: Mapped["Document"] = relationship(back_populates="doc_annotations")

    def __repr__(self) -> str:
        return f"DocAnnotation(id: {self.id}, type: {self.annotation_type}, value: {self.annotation_value})"

    def to_dict(self) -> dict:
        return { 
            "id": self.id,
            "document_id": self.document_id,
            "annotation_type": self.annotation_type,
            "annotation_value": self.annotation_value
        }
    

class SentAnnotation(Base):
    __tablename__ = "sent_annotations"
    id: Mapped[str] = mapped_column(String, primary_key=True) # sentid_annotationType
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    sentence_id: Mapped[str] = mapped_column(ForeignKey("sentences.id"))
    annotation_type: Mapped[str] = mapped_column(String)
    annotation_value: Mapped[str] = mapped_column(String)

    sentence: Mapped["Sentence"] = relationship(back_populates="sent_annotations")

    def __repr__(self) -> str:
        return f"DocAnnotation(id: {self.id}, type: {self.annotation_type}, value: {self.annotation_value})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sentence_id": self.sentence_id,
            "annotation_type": self.annotation_type,
            "annotation_value": self.annotation_value
        }