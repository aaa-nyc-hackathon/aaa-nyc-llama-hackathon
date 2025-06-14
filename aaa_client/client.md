# AthletIQ Client

This document provides an overview of the client application structure and setup instructions.

## Project Structure

```
aaa_client/
├── app/                    # Next.js app directory (pages and layouts)
├── components/            # Reusable React components
│   ├── ui/               # UI components (buttons, forms, etc.)
│   └── ...              # Other component categories
├── hooks/                # Custom React hooks
├── lib/                  # Utility functions and shared logic
├── public/              # Static assets
├── styles/              # Global styles and Tailwind configuration
├── .next/               # Next.js build output (generated)
├── node_modules/        # Dependencies (generated)
├── package.json         # Project dependencies and scripts
├── tsconfig.json        # TypeScript configuration
├── tailwind.config.ts   # Tailwind CSS configuration
└── next.config.mjs      # Next.js configuration
```

## Technology Stack

- **Framework**: Next.js 15.2.4
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **Form Handling**: React Hook Form
- **Validation**: Zod
- **State Management**: React Hooks
- **Animation**: Framer Motion

## Prerequisites

Before you begin, ensure you have the following installed:

- Node.js (Latest LTS version recommended)
- pnpm (Package manager)

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd aaa-nyc-llama-hackathon/aaa_client
   ```

2. **Install dependencies**

   ```bash
   pnpm install
   ```

3. **Start the development server**
   ```bash
   pnpm dev
   ```
   The application will be available at `http://localhost:3000`

## Available Scripts

- `pnpm dev` - Start the development server
- `pnpm build` - Build the application for production
- `pnpm start` - Start the production server
- `pnpm lint` - Run ESLint to check code quality

## Key Features

- Modern React with Next.js 15
- Type-safe development with TypeScript
- Responsive design with Tailwind CSS
- Accessible UI components from Radix UI
- Form validation with Zod
- Smooth animations with Framer Motion

## Development Guidelines

1. **Component Structure**

   - Place reusable UI components in `components/ui/`
   - Keep components small and focused
   - Use TypeScript for type safety

2. **Styling**

   - Use Tailwind CSS for styling
   - Follow the design system defined in `tailwind.config.ts`
   - Maintain consistent spacing and colors

3. **State Management**

   - Use React hooks for local state
   - Keep state as close as possible to where it's used
   - Use custom hooks for shared logic

4. **Code Quality**
   - Run `pnpm lint` before committing
   - Follow TypeScript best practices
   - Write meaningful component and function names

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Clear the `.next` directory and node_modules:
   ```bash
   rm -rf .next node_modules
   pnpm install
   ```
3. Check the console for error messages
4. Verify your Node.js version is compatible

## Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Radix UI Documentation](https://www.radix-ui.com/docs)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)
