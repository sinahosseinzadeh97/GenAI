from fastapi import APIRouter, HTTPException
from app.core.database import DBSession
from app import models, schemas

router = APIRouter()

@router.get("/workflows/{workflow_id}", response_model=schemas.WorkflowOut)
async def get_workflow(workflow_id: int):
    with DBSession() as db:
        wf = db.get(models.Workflow, workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return schemas.WorkflowOut(id=wf.id, type=wf.type, status=wf.status, payload=wf.payload, result=wf.result)