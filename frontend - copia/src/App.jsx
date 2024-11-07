import React from 'react';
import { ChakraProvider, Box, VStack, Heading, Container } from "@chakra-ui/react";
import MovieCarousel from './components/MovieList';
import ChatInterface from './components/ChatInterface';

const App = () => {
  return (
    <ChakraProvider>
      <Box minH="100vh" w="100vw" bg="gray.50" overflowX="hidden">
        <Container maxW="100%" p={0}>
          <VStack spacing={0} align="stretch">
            <Box bg="blue.600" py={4} px={8}>
              <Heading 
                as="h1" 
                size="2xl" 
                textAlign="center" 
                color="white"
                fontWeight="bold"
              >
                CineExplorer
              </Heading>
            </Box>
            <MovieCarousel />
            <Box px={8} py={6}>
              <ChatInterface />
            </Box>
          </VStack>
        </Container>
      </Box>
    </ChakraProvider>
  );
};

export default App;