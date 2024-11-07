import React, { useState, useEffect } from 'react';
import { Box, Flex, Image, Text, IconButton, useColorModeValue } from "@chakra-ui/react";
import { ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';

const MovieList = () => {
  const [movies, setMovies] = useState([]);
  const [scrollPosition, setScrollPosition] = useState(0);

  const moviePosters = {
    'Ted': 'https://m.media-amazon.com/images/M/MV5BMTQ1OTU0ODcxMV5BMl5BanBnXkFtZTcwOTMxNTUwOA@@._V1_.jpg',
    'Rush Hour': 'https://m.media-amazon.com/images/M/MV5BYWM2NDZmYmYtNzlmZC00M2MyLWJmOGUtMjhiYmQ2OGU1YTE1L2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_.jpg',
    'Rush Hour 2': 'https://m.media-amazon.com/images/I/71hetNUrA1S._AC_SL1347_.jpg',
    'Passengers': 'https://m.media-amazon.com/images/M/MV5BZjk4ZTMwMTYtOTk1NC00OTA0LWFhMGYtZTBjMzViMDY2YWZjXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg',
    'Oblivion': 'https://m.media-amazon.com/images/M/MV5BMTQwMDY0MTA4MF5BMl5BanBnXkFtZTcwNzI3MDgxOQ@@._V1_.jpg',
    'Limitless': 'https://m.media-amazon.com/images/M/MV5BYmViZGM0MGItZTdiYi00ZDU4LWIxNDYtNTc1NWQ5Njc2N2YwXkEyXkFqcGdeQXVyNDk3NzU2MTQ@._V1_.jpg',
    'American Pie': 'https://m.media-amazon.com/images/M/MV5BMTg3ODY5ODI1NF5BMl5BanBnXkFtZTgwMTkxNTYxMTE@._V1_.jpg',
    'Godfather Part II': 'https://m.media-amazon.com/images/M/MV5BMWMwMGQzZTItY2JlNC00OWZiLWIyMDctNDk2ZDQ2YjRjMWQ0XkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_.jpg',
    'Dune': 'https://m.media-amazon.com/images/I/71XRdOqbqVL._SL1081_.jpg',
    'Dune Part One': 'https://m.media-amazon.com/images/M/MV5BN2FjNmEyNWMtYzM0ZS00NjIyLTg5YzYtYThlMGVjNzE1OGViXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_FMjpg_UX1000_.jpg',
  };

  useEffect(() => {
    const fetchMovies = async () => {
      try {
        const response = await axios.get(`${import.meta.env.VITE_API_URL}/movies`);
        setMovies(response.data);
      } catch (error) {
        console.error('Error fetching movies:', error);
      }
    };
    fetchMovies();
  }, []);

  const scroll = (direction) => {
    const container = document.getElementById('movie-carousel');
    const scrollAmount = direction === 'left' ? -220 : 220;
    container.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    setScrollPosition(container.scrollLeft + scrollAmount);
  };

  return (
    <Box 
      position="relative" 
      bg="gray.900"
      py={4}
      width="100%"
      overflow="hidden"
    >
      <Flex 
        id="movie-carousel"
        overflowX="hidden"
        position="relative"
        scrollBehavior="smooth"
        px={16}
        gap={6}
        width="calc(100% + 32px)"
        ml="-16px"
      >
        {movies.map((movie, index) => (
          <Box
            key={`${movie.title}-${index}`}
            flexShrink={0}
            width="180px"
            borderRadius="lg"
            overflow="hidden"
            bg="gray.800"
            transition="transform 0.2s"
            _hover={{ transform: 'scale(1.05)' }}
          >
            <Box position="relative" width="180px" height="270px">
              <Image
                src={moviePosters[movie.title]}
                alt={movie.title}
                width="100%"
                height="100%"
                objectFit="cover"
                fallbackSrc="/placeholder.svg?height=270&width=180"
              />
            </Box>
            <Box p={4}>
              <Text color="white" fontWeight="bold" fontSize="md" whiteSpace="normal">
                {movie.title}
              </Text>
              <Text color="gray.400" fontSize="sm">
                Guión disponible
              </Text>
            </Box>
          </Box>
        ))}
      </Flex>
      <IconButton
        aria-label="Scroll left"
        icon={<ChevronLeft size={20} />}
        position="absolute"
        left={2}
        top="50%"
        transform="translateY(-50%)"
        onClick={() => scroll('left')}
        isDisabled={scrollPosition <= 0}
        colorScheme="whiteAlpha"
        rounded="full"
        size="sm"
        zIndex={2}
      />
      <IconButton
        aria-label="Scroll right"
        icon={<ChevronRight size={20} />}
        position="absolute"
        right={2}
        top="50%"
        transform="translateY(-50%)"
        onClick={() => scroll('right')}
        colorScheme="whiteAlpha"
        rounded="full"
        size="sm"
        zIndex={2}
      />
    </Box>
  );
};

export default MovieList;