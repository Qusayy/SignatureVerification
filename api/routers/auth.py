"""Employee authentication."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.db import get_session
from api.models.tables import Employee
from api.schemas import EmployeeOut, TokenResponse
from api.security.auth import authenticate, create_access_token, current_employee
from api.settings import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
) -> TokenResponse:
    employee = authenticate(session, form.username, form.password)
    if employee is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=create_access_token(employee),
        expires_in_minutes=get_settings().jwt_ttl_minutes,
        employee=EmployeeOut.model_validate(employee),
    )


@router.get("/me", response_model=EmployeeOut)
def me(employee: Employee = Depends(current_employee)) -> EmployeeOut:
    return EmployeeOut.model_validate(employee)
