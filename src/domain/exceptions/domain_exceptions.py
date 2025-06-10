
class DomainException(Exception):
    """Base exception for all domain-specific errors."""
    pass

class NotFoundException(DomainException):
    """Raised when a domain entity or aggregate is not found."""
    def __init__(self, entity_name: str, entity_id: str):
        self.entity_name = entity_name
        self.entity_id = entity_id
        super().__init__(f"{entity_name} with ID '{entity_id}' not found.")

class InvalidOperationException(DomainException):
    """Raised when an operation is invalid in the current domain state."""
    pass

class DataValidationException(DomainException):
    """Raised when data provided does not conform to domain rules."""
    pass