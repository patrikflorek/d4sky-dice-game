"""Module for displaying dice and numbers in ASCII art format.

This module provides functionality to:
1. Display dice faces with dots (1-6)
2. Display numbers in ASCII art format (0-9)
3. Combine dice and numbers for game state visualization

The display uses Unicode characters and colorama for colored output.

Example:
    >>> display = DiceDisplay()
    >>> display.display_position(6, 21)  # Shows dice face 6 and score 21
"""

from dataclasses import dataclass, field
from typing import Dict, List, ClassVar, Optional
from colorama import Fore, Style, init
import sys

# Initialize colorama for cross-platform colored output
init(autoreset=True)

@dataclass
class DisplayConfig:
    """Configuration for display appearance."""
    dot_char: str = '●'
    border_color: str = Fore.CYAN
    content_color: str = Fore.GREEN
    spacing: int = 2
    border_width: int = 30
    show_border: bool = True

@dataclass
class DiceDisplay:
    """Displays dice faces and numbers in ASCII art format."""
    
    # Configuration
    config: DisplayConfig = field(default_factory=DisplayConfig)
    
    # Internal state
    face_lines: Dict[int, List[str]] = field(init=False)
    num_lines: Dict[int, List[str]] = field(init=False)
    
    # Constants
    VALID_FACES: ClassVar[range] = range(1, 7)
    VALID_DIGITS: ClassVar[range] = range(10)
    LINE_HEIGHT: ClassVar[int] = 5
    
    def __post_init__(self):
        """Initialize the ASCII art patterns for dice faces and numbers."""
        self.face_lines = self._init_face_lines()
        self.num_lines = self._init_num_lines()
    
    def _init_face_lines(self) -> Dict[int, List[str]]:
        """Initialize ASCII art patterns for dice faces.
        
        Returns:
            Dictionary mapping face values to their ASCII art representations
        """
        dot = self.config.dot_char
        face_lines = {
            1: ["+-------+",
                f"       ",
                f"   {dot}   ",
                f"       ",
                "+-------+"],
                
            2: ["+-------+",
                f" {dot}     ",
                f"       ",
                f"     {dot} ",
                "+-------+"],
                
            3: ["+-------+",
                f" {dot}     ",
                f"   {dot}   ",
                f"     {dot} ",
                "+-------+"],
                
            4: ["+-------+",
                f" {dot}   {dot} ",
                f"       ",
                f" {dot}   {dot} ",
                "+-------+"],
                
            5: ["+-------+",
                f" {dot}   {dot} ",
                f"   {dot}   ",
                f" {dot}   {dot} ",
                "+-------+"],
                
            6: ["+-------+",
                f" {dot}   {dot} ",
                f" {dot}   {dot} ",
                f" {dot}   {dot} ",
                "+-------+"]
        }
        
        # Add vertical borders
        for face in face_lines:
            for i in range(1, len(face_lines[face])-1):
                face_lines[face][i] = '|' + face_lines[face][i] + '|'
        
        return face_lines

    def _init_num_lines(self) -> Dict[int, List[str]]:
        """Initialize ASCII art patterns for numbers.
        
        Returns:
            Dictionary mapping digits to their ASCII art representations
        """
        return {
            0: [" ██████ ",
                "██  ████",
                "██ ██ ██",
                "████  ██",
                " ██████ "],
                
            1: [" ██     ",
                "███     ",
                " ██     ",
                " ██     ",
                " ██     "],
                
            2: ["███████ ",
                "     ██ ",
                " █████  ",
                "██      ",
                "███████ "],
                
            3: ["██████  ",
                "     ██ ",
                " █████  ",
                "     ██ ",
                "██████  "],
                
            4: ["██   ██ ",
                "██   ██ ",
                "███████ ",
                "     ██ ",
                "     ██ "],
                
            5: ["███████ ",
                "██      ",
                "███████ ",
                "     ██ ",
                "███████ "],
                
            6: [" ██████ ",
                "██      ",
                "███████ ",
                "██    ██",
                " ██████ "],
                
            7: ["███████ ",
                "     ██ ",
                "    ██  ",
                "   ██   ",
                "   ██   "],
                
            8: [" █████  ",
                "██   ██ ",
                " █████  ",
                "██   ██ ",
                " █████  "],
                
            9: [" █████  ",
                "██   ██ ",
                " ██████ ",
                "     ██ ",
                " █████  "]
        }

    def display_position(self, face: int, suma: int, *, show_border: Optional[bool] = None) -> None:
        """Display the current game position with dice face and sum.
        
        Args:
            face: Current dice face value (1-6)
            suma: Current sum value (0-99)
            show_border: Override config.show_border for this display
            
        Raises:
            ValueError: If face or suma are outside valid ranges
        """
        # Validate inputs
        if face not in self.VALID_FACES:
            raise ValueError(f"Invalid face value: {face}. Must be between 1 and 6")
        if not (0 <= suma <= 99):
            raise ValueError(f"Invalid sum value: {suma}. Must be between 0 and 99")
            
        # Convert sum to string and validate digits
        str_suma = str(suma)
        if not all(int(d) in self.VALID_DIGITS for d in str_suma):
            raise ValueError(f"Invalid digits in sum: {suma}")
        
        # Build display lines
        lines = []
        for li in range(self.LINE_HEIGHT):    
            # Start with dice face
            line = self.face_lines[face][li]
            
            # Add spacing
            line += self.config.spacing * " "
            
            # Add each digit
            for digit in str_suma:
                line += self.num_lines[int(digit)][li]
                line += self.config.spacing * " "
            
            lines.append(line)
        
        # Display with optional border
        show = self.config.show_border if show_border is None else show_border
        if show:
            self._print_with_border(lines)
        else:
            self._print_content(lines)
    
    def _print_with_border(self, lines: List[str]) -> None:
        """Print content with decorative border.
        
        Args:
            lines: Content lines to display
        """
        border = self.config.border_width * "*"
        print(f"{self.config.border_color}{border}\n{border}")
        print()
        self._print_content(lines)
        print()
        print(f"{self.config.border_color}{border}\n{border}")
    
    def _print_content(self, lines: List[str]) -> None:
        """Print the main content lines.
        
        Args:
            lines: Content lines to display
        """
        for line in lines:
            print(f"{self.config.content_color}{line}")
