from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.estado import EstadoChat
from app.graph.nodos import detectar_tema_y_contexto, explicar, verificar, evaluar_verificacion

builder = StateGraph(EstadoChat)
builder.add_node("detectar_tema_y_contexto", detectar_tema_y_contexto)
builder.add_node("explicar", explicar)
builder.add_node("verificar", verificar)
builder.add_node("evaluar_verificacion", evaluar_verificacion)

builder.set_entry_point("detectar_tema_y_contexto")
builder.add_edge("detectar_tema_y_contexto", "explicar")
builder.add_edge("explicar", "verificar")
builder.add_edge("verificar", "evaluar_verificacion")
builder.add_edge("evaluar_verificacion", END)

checkpointer = MemorySaver()
grafo = builder.compile(checkpointer=checkpointer)